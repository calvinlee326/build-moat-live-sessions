"""MCP server for the task scheduler.

Run as a stdio MCP server:
    python -m app.mcp_server

Or test with the inspector:
    npx @modelcontextprotocol/inspector python -m app.mcp_server

MCP (Model Context Protocol) is a standard for exposing tools to LLMs.
The server communicates over stdin/stdout — that's why it appears to "hang"
when run directly. It's waiting for JSON-RPC messages from a client.
"""

import asyncio
import json
from datetime import datetime

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool
from sqlalchemy.orm import Session

from .database import Base, SessionLocal, engine
from .models import Job
from .scheduler import get_time_bucket, start_scheduler


# ===================================================================
# Tool handlers — pure business logic, sync, take a DB Session
# ===================================================================


def handle_create_task(db: Session, *, description: str, scheduled_at: str) -> dict:
    """Create a new scheduled job."""
    # fromisoformat parses strings like "2026-05-15T10:00:00" into a datetime object
    dt = datetime.fromisoformat(scheduled_at)
    job = Job(
        description=description,
        scheduled_at=dt,
        # Compute the time bucket at creation time so the watcher query is fast
        time_bucket=get_time_bucket(dt),
    )
    db.add(job)
    db.commit()
    # refresh() re-reads the row so we get the auto-generated id
    db.refresh(job)
    return {"job_id": job.id, "status": job.status, "scheduled_at": str(job.scheduled_at)}


def handle_get_status(db: Session, *, job_id: int) -> dict:
    """Get the status of a scheduled job."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if job is None:
        return {"error": f"Job {job_id} not found"}
    return {
        "job_id": job.id,
        "description": job.description,
        "status": job.status,
        "scheduled_at": str(job.scheduled_at),
        "result": job.result,
    }


def handle_list_tasks(db: Session) -> dict:
    """List all scheduled jobs."""
    jobs = db.query(Job).order_by(Job.scheduled_at.desc()).all()
    return {
        # List comprehension builds a list of dicts from the SQLAlchemy row objects
        "jobs": [
            {
                "job_id": j.id,
                "description": j.description,
                "status": j.status,
                "scheduled_at": str(j.scheduled_at),
            }
            for j in jobs
        ]
    }


def handle_cancel_task(db: Session, *, job_id: int) -> dict:
    """Cancel a scheduled job."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if job is None:
        return {"error": f"Job {job_id} not found"}
    # Can't cancel what's already finished — those states are terminal
    if job.status in ("completed", "failed"):
        return {"error": f"Cannot cancel job in '{job.status}' state"}
    job.status = "cancelled"
    db.commit()
    return {"job_id": job.id, "status": "cancelled"}


# ===================================================================
# Tool definitions — what Claude / MCP client sees
# (pre-filled — boilerplate for MCP discovery, not the focus)
# ===================================================================

# Each Tool entry tells the LLM: "this tool exists, here's what it does,
# and here's the JSON schema for its inputs." The LLM uses this to decide
# which tool to call and how to format its arguments.
TOOL_DEFINITIONS: list[Tool] = [
    Tool(
        name="task.create",
        description="Schedule a new task for future execution",
        inputSchema={
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "What the task should do",
                },
                "scheduled_at": {
                    "type": "string",
                    "format": "date-time",
                    "description": "When to run, ISO 8601 format (e.g. 2026-05-03T10:00:00)",
                },
            },
            "required": ["description", "scheduled_at"],
        },
    ),
    Tool(
        name="task.list",
        description="List all scheduled tasks",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="task.status",
        description="Get the status of a scheduled task by job_id",
        inputSchema={
            "type": "object",
            "properties": {
                "job_id": {"type": "integer", "description": "The job ID returned by task.create"},
            },
            "required": ["job_id"],
        },
    ),
    Tool(
        name="task.cancel",
        description="Cancel a scheduled task that hasn't completed yet",
        inputSchema={
            "type": "object",
            "properties": {
                "job_id": {"type": "integer", "description": "The job ID to cancel"},
            },
            "required": ["job_id"],
        },
    ),
]


# ===================================================================
# Registry pattern — name to handler dispatch
# ===================================================================

# A dict maps tool names → handler functions.
# Why not if/elif? Adding a 5th tool is one line here instead of touching
# the dispatcher logic. Easier to read, test, and extend.
TOOL_REGISTRY: dict = {
    "task.create": handle_create_task,
    "task.list":   handle_list_tasks,
    "task.status": handle_get_status,
    "task.cancel": handle_cancel_task,
}


def route_tool_call(tool_name: str, arguments: dict, db: Session) -> dict:
    """Single dispatch point — look up handler in TOOL_REGISTRY and call it.

    Kept sync so handlers can use plain SQLAlchemy without async ceremony.
    The async wrapper below runs this in a thread via asyncio.to_thread().
    """
    handler = TOOL_REGISTRY.get(tool_name)
    if handler is None:
        return {"error": f"Unknown tool: {tool_name}"}
    # **arguments unpacks the dict as keyword arguments to match each handler's signature
    return handler(db, **arguments)


# ===================================================================
# MCP server wiring — boilerplate, do not modify
# ===================================================================

server: Server = Server("task-scheduler")


@server.list_tools()
async def list_tools() -> list[Tool]:
    # Called by the MCP client on connect to discover available tools
    return TOOL_DEFINITIONS


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Async entry point for every tool call from the LLM.

    asyncio.to_thread() runs the sync handler in a thread pool so the
    event loop isn't blocked while SQLAlchemy does DB work.
    TextContent wraps the result as a JSON string the MCP client can read.
    """
    db = SessionLocal()
    try:
        result = await asyncio.to_thread(route_tool_call, name, arguments or {}, db)
    finally:
        db.close()
    return [TextContent(type="text", text=json.dumps(result, default=str, ensure_ascii=False))]


# ===================================================================
# Entry point — `python -m app.mcp_server`
# ===================================================================


async def main() -> None:
    # Create DB tables if they don't exist yet (safe to call repeatedly)
    Base.metadata.create_all(bind=engine)
    # Launch watcher + worker threads in the background
    start_scheduler()

    # stdio_server() connects the MCP protocol to stdin/stdout
    # The server will block here reading messages until the process is killed
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
