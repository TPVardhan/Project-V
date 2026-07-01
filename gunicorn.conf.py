"""
Gunicorn configuration for Project-V.

KEY FIX: APScheduler's BackgroundScheduler drives jobs from a background
thread.  Python's os.fork() copies memory but does NOT clone threads, so
the scheduler thread that started in the Gunicorn master is dead in every
worker process.  The worker therefore inherits a scheduler object whose
.running flag is True but whose execution thread is gone — check_automations
never fires.

The post_fork hook below restarts the scheduler inside each worker so it
actually runs where requests are served.
"""

workers = 1
timeout = 120   # allow slow outbound HTTP checks without worker kill


def post_fork(server, worker):
    from app import scheduler
    # Shut down the inherited-but-dead scheduler, then restart it cleanly
    # inside this worker process.
    try:
        if scheduler.running:
            scheduler.shutdown(wait=False)
    except Exception:
        pass
    scheduler.start()
    server.log.info("APScheduler restarted in worker %s", worker.pid)
