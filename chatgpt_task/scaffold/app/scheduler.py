import queue
import threading
import time
from datetime import datetime

from sqlalchemy.orm import Session

from .database import SessionLocal
from .models import Job, _utcnow

# In-memory queue connecting the watcher and worker threads.
# In production this would be SQS or Redis so jobs survive restarts.
# queue.Queue is thread-safe — no manual locking needed.
job_queue: queue.Queue[int] = queue.Queue()


def get_time_bucket(scheduled_at: datetime) -> str:
    """Convert a datetime to an hourly bucket string (e.g. '2026051512').

    Why bucket by hour? The DB index is on (time_bucket, status).
    Without buckets, every watcher tick scans the whole jobs table.
    With buckets, it only reads the current hour's slice — O(1) regardless
    of how many jobs exist in past hours.
    """
    # strftime format: Year(4) + Month(2) + Day(2) + Hour(2)
    # Zero-padding means lexicographic order == chronological order,
    # so "2026051511" < "2026051512" works correctly with `<=`.
    return scheduled_at.strftime("%Y%m%d%H")


def find_due_jobs(current_time: datetime, db: Session) -> list[Job]:
    """Return all pending jobs whose scheduled time has passed.

    Uses `time_bucket <= current_bucket` (not ==) so jobs from earlier
    hours that were missed (e.g. server was down) are still picked up.
    """
    current_bucket = get_time_bucket(current_time)
    return (
        db.query(Job)
        .filter(
            # Partition filter: only look at current and past hour buckets
            Job.time_bucket <= current_bucket,
            # Fine-grained check: scheduled time has actually passed
            Job.scheduled_at <= current_time,
            # Only pick up jobs that haven't been started or cancelled yet
            Job.status == "pending",
        )
        .all()
    )


def watcher_loop(interval: int = 10):
    """Watcher runs forever, scanning the DB every `interval` seconds.

    Responsibility: find due jobs and push their IDs to the queue.
    It does NOT execute jobs — that's the worker's job.
    Keeping these separate means a slow job never blocks the next scan.
    """
    while True:
        db = SessionLocal()
        try:
            now = _utcnow()
            due_jobs = find_due_jobs(now, db)
            for job in due_jobs:
                # Mark as queued immediately so a second watcher tick won't re-enqueue it
                job.status = "queued"
                db.commit()
                # Only the job ID goes on the queue — worker re-fetches the full row
                job_queue.put(job.id)
        finally:
            db.close()
        time.sleep(interval)


def worker_loop():
    """Worker runs forever, pulling job IDs from the queue and executing them.

    job_queue.get() blocks until a job is available — no busy waiting/polling.
    task_done() signals queue.join() callers that this item is processed.
    """
    while True:
        job_id = job_queue.get()  # blocks here until the watcher enqueues something
        db = SessionLocal()
        try:
            job = db.query(Job).filter(Job.id == job_id).first()
            # Guard: job may have been cancelled between queueing and execution
            if job is None or job.status == "cancelled":
                continue

            job.status = "running"
            db.commit()

            # Simulate execution — in production this would call an LLM or external API
            job.result = f"Executed: {job.description}"
            job.status = "completed"
            db.commit()
        except Exception as e:
            job.status = "failed"
            job.result = str(e)
            db.commit()
        finally:
            db.close()
            job_queue.task_done()  # must always call this, even on error


def start_scheduler():
    """Launch watcher and worker as background daemon threads.

    daemon=True means these threads die automatically when the main process exits.
    Without it, the server would hang on shutdown waiting for the threads to stop.
    """
    watcher = threading.Thread(target=watcher_loop, daemon=True)
    worker = threading.Thread(target=worker_loop, daemon=True)
    watcher.start()
    worker.start()
