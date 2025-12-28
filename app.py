import os
import sqlite3
from datetime import datetime

import requests
from requests.exceptions import RequestException
from flask import Flask, render_template, request, jsonify, g
from dotenv import load_dotenv
from openai import OpenAI
from apscheduler.schedulers.background import BackgroundScheduler
import json

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "project-v-secret-key-change-me")

client = OpenAI(api_key=OPENAI_API_KEY)

scheduler = BackgroundScheduler()
scheduler.start()

# ---------- SIMPLE TEXT HELPERS (INTENT MATCHING) ----------

STOPWORDS = {
    "the", "a", "an", "of", "for", "to", "and", "or", "in", "on",
    "at", "by", "with", "from", "is", "are", "was", "were",
    "this", "that", "these", "those"
}

def text_to_tokens(text: str):
    if not text:
        return set()
    text = text.lower()
    for ch in [",", ".", ":", ";", "!", "?", "(", ")", "[", "]", "{", "}", "-", "–", "—", "/", "\\", "|"]:
        text = text.replace(ch, " ")
    raw_tokens = text.split()
    tokens = {t for t in raw_tokens if t and t not in STOPWORDS}
    return tokens

def keyword_matches_page(keyword: str, page_text: str, threshold: float = 0.6) -> bool:
    """
    Foolproof keyword matching that works for ANY website:
    - Movies, exams, products, anything
    - Handles partial matches (e.g., "Anaconda 2025" matches even if year is separate)
    - Gives bonus points for intent words (release, notify, available, etc.)
    """
    if not keyword or not page_text:
        return False

    kw_tokens = text_to_tokens(keyword)
    if not kw_tokens:
        return False
    
    page_tokens = text_to_tokens(page_text)

    overlap = kw_tokens.intersection(page_tokens)
    
    if not overlap:
        return False

    ratio = len(overlap) / len(kw_tokens)

    # EXPANDED important tokens for ANY use case (exams + movies + products)
    important = {
        # Exams
        "ssc", "upsc", "cgl", "chsl", "tier", "mains", "prelims",
        "result", "results", "notification", "interview", "schedule",
        "admit", "card", "hall", "ticket",
        # Movies & Entertainment
        "release", "released", "now", "showing", "available", "watch",
        "streaming", "premiere", "launch", "book", "tickets",
        # General Commerce
        "buy", "order", "alert", "notify", "price", "stock",
        "in stock", "available", "sold", "offer"
    }
    
    important_overlap = important.intersection(overlap)

    # Any 4-digit number is probably a year (2025, etc.)
    has_year = any(t.isdigit() and len(t) == 4 for t in overlap)

    # FOOLPROOF: If we matched the main keyword + any intent word, trigger it
    # This works for: "Anaconda 2025" when page has "Anaconda" + "release"
    if important_overlap and ratio >= 0.4:
        return True

    # If main keyword present + year present, trigger (even without intent words)
    if has_year and ratio >= 0.4:
        return True

    # Otherwise require lower threshold (was 0.6, now 0.5 for real websites)
    return ratio >= 0.5


# ---------- DATABASE HELPERS ----------

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect("project_v.db", check_same_thread=False)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(error):
    db = g.pop("db", None)
    if db is not None:
        db.close()

def init_db():
    db = get_db()
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS automations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            description TEXT NOT NULL,
            url TEXT,
            keyword TEXT,
            frequency_minutes INTEGER DEFAULT 15,
            contact TEXT,
            status TEXT DEFAULT 'active',          -- active / completed / paused
            last_checked TEXT,
            last_status TEXT DEFAULT 'pending',    -- pending / not_found / triggered / error / http_xxx
            keywords_json TEXT
        )
        """
    )

    db.execute(
        """
        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            automation_id INTEGER,
            message TEXT NOT NULL,
            created_at TEXT NOT NULL,
            is_read INTEGER DEFAULT 0,
            FOREIGN KEY (automation_id) REFERENCES automations(id)
        )
        """
    )
    db.commit()

# ---------- OPENAI PARSER ----------

def parse_automation_instruction(instruction: str, user_freq: int | None = None):
    """
    Parse the natural language instruction into structured fields.
    If user_freq is provided, the model is allowed to suggest a value,
    but the caller can still override it with user_freq later.
    """
    freq_hint = f"User suggested frequency_minutes: {user_freq}." if user_freq else "No explicit user frequency given."

    prompt = f"""
You are a helper that converts ONE automation sentence into simple JSON for web-page checking.

Return ONLY JSON, no explanation, with exactly:
{{
  "site_domain": "example.com",
  "topic": "short description of what to watch for",
  "url": "https://example.com/page",
  "main_keyword": "the single most important phrase to detect",
  "backup_keywords": ["alternative phrase 1", "alternative phrase 2"],
  "frequency_minutes": 5
}}

Rules:
- "site_domain": official website domain if known from the text
  - UPSC -> "upsc.gov.in"
  - SSC -> "ssc.nic.in"
  - BookMyShow -> "in.bookmyshow.com"
  - If you are not sure, guess the most likely official domain.
- "topic": 5–15 words summarising what the user wants.
- "url": if a very likely page is obvious, give it; otherwise use "https://example.com".
- "main_keyword": phrase that appears (or is very likely) in HTML when event is true.
- "backup_keywords": 2–5 short variations (2–6 words).
- "frequency_minutes": use {freq_hint} If none, pick between 10 and 30.

User instruction:
"{instruction}"
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        content = resp.choices[0].message.content.strip()
        data = json.loads(content)

        site_domain = data.get("site_domain")
        topic = data.get("topic")
        url = data.get("url")
        main_keyword = data.get("main_keyword")
        backup_keywords = data.get("backup_keywords") or []
        freq = data.get("frequency_minutes", 15)

        if not isinstance(freq, int):
            freq = 15
        if not isinstance(backup_keywords, list):
            backup_keywords = []

        # UPSC special case
        if site_domain and site_domain.lower() == "upsc.gov.in":
            topic_text = (topic or "").lower()
            if "interview" in topic_text:
                precise = "Interview Schedule: Civil Services (Main) Examination, 2025"
                main_keyword = precise
                if precise not in backup_keywords:
                    backup_keywords.append(precise)

        return {
            "site_domain": site_domain,
            "topic": topic,
            "url": url,
            "main_keyword": main_keyword,
            "backup_keywords": backup_keywords,
            "frequency_minutes": freq,
        }
    except Exception:
        return {
            "site_domain": None,
            "topic": instruction,
            "url": None,
            "main_keyword": instruction,
            "backup_keywords": [],
            "frequency_minutes": user_freq if isinstance(user_freq, int) else 15,
        }

def resolve_initial_url(site_domain: str, topic: str) -> str | None:
    if not site_domain:
        return None

    domain = site_domain.lower()
    topic_text = (topic or "").lower()

    if "upsc.gov.in" in domain and "interview" in topic_text:
        return "https://upsc.gov.in/exams-related-info/interview-schedule"

    return f"https://{domain.rstrip('/')}"

# ---------- REAL HTTP CHECKER + NOTIFICATIONS ----------

def create_notification(db, automation_row, message):
    now_iso = datetime.utcnow().isoformat()
    db.execute(
        """
        INSERT INTO notifications (automation_id, message, created_at, is_read)
        VALUES (?, ?, ?, 0)
        """,
        (automation_row["id"], message, now_iso),
    )

def _fetch_url_with_retry(url: str, timeout: int = 8, max_tries: int = 2):
    """
    Try fetching the URL up to max_tries times before giving up.
    Returns (resp, error_flag). error_flag is True only if all tries failed.
    """
    last_exc = None
    for attempt in range(max_tries):
        try:
            resp = requests.get(
                url,
                timeout=timeout,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; Project-V/1.0)"
                },
            )
            return resp, False
        except RequestException as e:
            last_exc = e
    # All attempts failed
    return last_exc, True

def check_automations():
    print("CHECK_AUTOMATIONS RUNNING")

    with app.app_context():
        db = get_db()
        # Only check active tasks; completed/paused are ignored
        rows = db.execute(
            "SELECT * FROM automations WHERE status = 'active'"
        ).fetchall()

        now_iso = datetime.utcnow().isoformat()

        for row in rows:
            url = row["url"]
            single_keyword = row["keyword"]
            keywords_json = row["keywords_json"]

            all_keywords = []

            if keywords_json:
                try:
                    payload = json.loads(keywords_json)
                    mk = payload.get("main_keyword")
                    bks = payload.get("backup_keywords") or []
                    if mk:
                        all_keywords.append(mk)
                    for k in bks:
                        if isinstance(k, str):
                            all_keywords.append(k)
                except Exception:
                    pass

            if not all_keywords and single_keyword:
                all_keywords.append(single_keyword)

            if not url or not all_keywords:
                continue

            matched_keyword = None

            try:
                # Fetch with retry (helps when UPSC/SSC is briefly slow)
                resp, failed = _fetch_url_with_retry(url, timeout=8, max_tries=2)

                if failed:
                    last_status = "error"
                    last_checked = f"Network error at {now_iso}"
                else:
                    if isinstance(resp, requests.Response) and resp.status_code == 200:
                        body_text = resp.text
                        body_lower = body_text.lower()

                        found = False

                        # 1) Prefer direct substring match of intent phrases
                        for k in all_keywords:
                            if not isinstance(k, str):
                                continue
                            key_lower = k.lower().strip()
                            if key_lower and key_lower in body_lower:
                                found = True
                                matched_keyword = k
                                break

                        # 2) If not found, use smarter token overlap
                        if not found:
                            is_example = "example.com" in (url or "").lower()
                            # More lenient for example.com demo tasks
                            demo_threshold = 0.3 if is_example else 0.6

                            for k in all_keywords:
                                if keyword_matches_page(
                                    k, body_text, threshold=demo_threshold
                                ):
                                    found = True
                                    matched_keyword = k
                                    break

                        if found:
                            last_status = "triggered"
                            last_checked = f"Keyword FOUND ({matched_keyword}) at {now_iso}"
                        else:
                            last_status = "not_found"
                            last_checked = f"Checked at {now_iso}, not found"
                    else:
                        status_code = getattr(resp, "status_code", "unknown")
                        last_status = f"http_{status_code}"
                        last_checked = f"HTTP error {status_code} at {now_iso}"

            except Exception:
                last_status = "error"
                last_checked = f"Unexpected error at {now_iso}"

            previous_state = row["status"]
            previous_last_status = row["last_status"]

            # If triggered: send one notification and mark automation as completed (stop future checks)
            if last_status == "triggered" and previous_state == "active":
                msg = f"Automation #{row['id']}: keyword found"
                detail_keyword = matched_keyword if matched_keyword else all_keywords[0]
                detail = f"{detail_keyword} @ {url}"
                create_notification(db, row, f"{msg}||{detail}")

                db.execute(
                    """
                    UPDATE automations
                    SET status = 'completed',
                        last_status = ?,
                        last_checked = ?
                    WHERE id = ?
                    """,
                    (last_status, last_checked, row["id"]),
                )
            else:
                # For the very first run (still pending), hide transient errors/http_xxx;
                # only update last_checked so the UI stays as "pending" until a proper result.
                if previous_last_status == "pending" and last_status not in ("triggered", "not_found"):
                    db.execute(
                        """
                        UPDATE automations
                        SET last_checked = ?
                        WHERE id = ?
                        """,
                        (last_checked, row["id"]),
                    )
                else:
                    db.execute(
                        """
                        UPDATE automations
                        SET last_status = ?, last_checked = ?
                        WHERE id = ?
                        """,
                        (last_status, last_checked, row["id"]),
                    )

        db.commit()

scheduler.add_job(check_automations, "interval", minutes=1)

# ---------- ROUTES ----------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat")
def chat():
    first_question = request.args.get("q", "")
    return render_template("chat.html", first_question=first_question)


@app.route("/automation")
def automation_page():
    db = get_db()
    rows = db.execute(
        "SELECT * FROM automations ORDER BY id DESC"
    ).fetchall()
    return render_template("automation.html", automations=rows)

@app.route("/api/automation/create", methods=["POST"])
def api_create_automation():
    data = request.get_json() or {}
    instruction = (data.get("instruction") or "").strip()

    if not instruction:
        return jsonify({"success": False, "error": "instruction_required"}), 400

    # Optional overrides from user
    user_freq = data.get("frequency_minutes")
    try:
        if user_freq is not None:
            user_freq = int(user_freq)
            if user_freq < 1:
                user_freq = 1
    except (TypeError, ValueError):
        user_freq = None

    user_url = (data.get("target_url") or "").strip() or None
    extra_notes = (data.get("extra_notes") or "").strip() or None

    # Parse instruction (model can still propose its own frequency)
    parsed = parse_automation_instruction(instruction, user_freq=user_freq)

    site_domain = parsed.get("site_domain")
    topic = parsed.get("topic")

    resolved_url = resolve_initial_url(site_domain, topic)

    raw_url = parsed.get("url")
    if site_domain and site_domain.lower() == "upsc.gov.in":
        url = resolved_url
    else:
        url = raw_url or resolved_url

    # If user provided a URL explicitly, override everything
    if user_url:
        url = user_url

    main_keyword = parsed.get("main_keyword")
    backup_keywords = parsed.get("backup_keywords") or []
    freq = parsed.get("frequency_minutes", 15)

    # If user gave a frequency, override model suggestion
    if isinstance(user_freq, int):
        freq = user_freq

    print("NEW AUTOMATION PARSED:")
    print("  site_domain:", site_domain)
    print("  topic      :", topic)
    print("  url        :", url)
    print("  main_kw    :", main_keyword)
    print("  backups    :", backup_keywords)
    print("  freq       :", freq)
    print("  extra_notes:", extra_notes)

    keywords_payload = {
        "main_keyword": main_keyword,
        "backup_keywords": backup_keywords,
        "extra_notes": extra_notes,
    }
    keywords_json = json.dumps(keywords_payload, ensure_ascii=False)

    db = get_db()
    db.execute(
        """
        INSERT INTO automations
        (description, url, keyword, frequency_minutes, status, keywords_json)
        VALUES (?, ?, ?, ?, 'active', ?)
        """,
        (instruction, url, main_keyword, freq, keywords_json),
    )
    db.commit()

    return jsonify(
        {
            "success": True,
            "parsed": parsed,
        }
    )

# ---------- AUTOMATIONS API FOR TABLE ----------

@app.route("/api/automations", methods=["GET"])
def api_automations():
    db = get_db()
    rows = db.execute(
        """
        SELECT id, description, status, frequency_minutes,
               last_status, last_checked
        FROM automations
        ORDER BY id DESC
        """
    ).fetchall()

    items = []
    for r in rows:
        items.append(
            {
                "id": r["id"],
                "description": r["description"],
                "status": r["status"],
                "frequency_minutes": r["frequency_minutes"],
                "last_status": r["last_status"],
                "last_checked": r["last_checked"],
            }
        )
    return jsonify({"items": items})

# ---------- NOTIFICATIONS & DEBUG APIs ----------

@app.route("/api/notifications", methods=["GET"])
def api_notifications():
    db = get_db()
    rows = db.execute(
        """
        SELECT n.id, n.automation_id, n.message, n.created_at, n.is_read,
               a.description
        FROM notifications n
        LEFT JOIN automations a ON n.automation_id = a.id
        WHERE n.is_read = 0
        ORDER BY n.created_at DESC
        LIMIT 20
        """
    ).fetchall()
    return jsonify([dict(r) for r in rows])

@app.route("/api/automation/<int:auto_id>/details", methods=["GET"])
def api_automation_details(auto_id):
    db = get_db()
    row = db.execute(
        """
        SELECT id, description, url, keyword, frequency_minutes,
               status, last_status, last_checked, keywords_json
        FROM automations
        WHERE id = ?
        """,
        (auto_id,),
    ).fetchone()

    if row is None:
        return jsonify({"error": "not_found"}), 404

    return jsonify(
        {
            "id": row["id"],
            "description": row["description"],
            "url": row["url"],
            "keyword": row["keyword"],
            "frequency_minutes": row["frequency_minutes"],
            "status": row["status"],
            "last_status": row["last_status"],
            "last_checked": row["last_checked"],
            "keywords_json": row["keywords_json"],
        }
    )

@app.route("/api/notifications/mark-read", methods=["POST"])
def api_notifications_mark_read():
    db = get_db()
    db.execute("UPDATE notifications SET is_read = 1 WHERE is_read = 0")
    db.commit()
    return jsonify({"success": True})

# ---------- NEW: PAUSE/RESUME & DELETE APS ----------

@app.route("/api/automation/<int:auto_id>/toggle", methods=["POST"])
def api_automation_toggle(auto_id):
    """
    Toggle one automation between active and paused.
    Completed tasks stay completed and are not reactivated.
    """
    db = get_db()
    row = db.execute(
        "SELECT status FROM automations WHERE id = ?",
        (auto_id,),
    ).fetchone()
    if row is None:
        return jsonify({"error": "not_found"}), 404

    current = row["status"]

    if current == "completed":
        # Do not change completed tasks
        return jsonify(
            {
                "success": False,
                "status": current,
                "message": "completed task cannot be toggled",
            }
        ), 400

    new_status = "paused" if current == "active" else "active"

    db.execute(
        "UPDATE automations SET status = ? WHERE id = ?",
        (new_status, auto_id),
    )
    db.commit()
    return jsonify({"success": True, "status": new_status})

@app.route("/api/automation/<int:auto_id>", methods=["DELETE"])
def api_automation_delete(auto_id):
    """
    Delete one automation and its notifications.
    """
    db = get_db()
    db.execute("DELETE FROM notifications WHERE automation_id = ?", (auto_id,))
    db.execute("DELETE FROM automations WHERE id = ?", (auto_id,))
    db.commit()
    return jsonify({"success": True})

# ---------- CHAT API (for chat.html) ----------

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json() or {}
    message = (data.get("message") or "").strip()
    
    if not message:
        return jsonify({"error": "No message provided"}), 400
    
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": message}],
            temperature=0.7,
        )
        reply = resp.choices[0].message.content.strip()
        return jsonify({"reply": reply}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/new_chat", methods=["POST"])
def api_new_chat():
    # Clear conversation (placeholder for now - you can add session management later)
    return jsonify({"success": True}), 200




# ---------- MAIN ----------

if __name__ == "__main__":
    with app.app_context():
        init_db()
    print("🚀 Project‑V started with run-until-trigger-then-stop behavior.")
    app.run(debug=True)
# -----------------------

@app.route('/status')
@app.route('/docs')
@app.route('/support')
@app.route('/pricing')
@app.route('/changelog')
@app.route('/examples')
@app.route('/contact')
@app.route('/privacy')
@app.route('/terms')
def coming_soon():
    return render_template('coming_soon.html')

#----------------------------------------------------
@app.route('/api/notifications', methods=['GET'])
def get_notifications():
    try:
        notifications = Notification.query.order_by(Notification.created_at.desc()).limit(50).all()
        items = [{
            'id': n.id,
            'message': n.message,
            'created_at': n.created_at.strftime('%Y-%m-%d %H:%M:%S') if n.created_at else ''
        } for n in notifications]
        return jsonify(items), 200
    except Exception as e:
        print(f"Error in get_notifications: {e}")
        return jsonify([]), 200


@app.route('/api/notifications/mark-read', methods=['POST'])
def mark_notifications_read():
    try:
        Notification.query.update({'read': True})
        db.session.commit()
        return jsonify({'success': True}), 200
    except Exception as e:
        print(f"Error marking as read: {e}")
        return jsonify({'error': str(e)}), 500
#--------------------------------------------

