import sqlite3

conn = sqlite3.connect("project_v.db")
c = conn.cursor()

# Add a new column ONLY if it does not exist yet
try:
    c.execute("ALTER TABLE automations ADD COLUMN keywords_json TEXT")
    print("Column keywords_json added.")
except sqlite3.OperationalError as e:
    print("Maybe column already exists:", e)

conn.commit()
conn.close()
print("Done.")
