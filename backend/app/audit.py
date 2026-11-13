"""Structured, exportable audit events for research and security decisions."""

import csv
import io
import json
import logging
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
_events: List[Dict[str, Any]] = []
_lock = Lock()


def record_audit_event(event: str, *, session_id: Optional[str] = None, **details: Any) -> Dict[str, Any]:
    entry = {
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        **details,
    }
    with _lock:
        _events.append(entry)
        if len(_events) > 10000:
            del _events[: len(_events) - 10000]
    logger.info("audit_event=%s", json.dumps(entry, default=str, sort_keys=True))
    return entry


def get_audit_events(session_id: Optional[str] = None) -> List[Dict[str, Any]]:
    with _lock:
        events = list(_events)
    return [event for event in events if session_id is None or event.get("session_id") == session_id]


def export_audit_events(fmt: str, session_id: Optional[str] = None) -> tuple[str, str]:
    events = get_audit_events(session_id)
    if fmt == "json":
        return json.dumps(events, indent=2, default=str), "application/json"
    if fmt == "csv":
        output = io.StringIO()
        fields = sorted({key for event in events for key in event})
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for event in events:
            writer.writerow({key: json.dumps(event.get(key), default=str) if isinstance(event.get(key), (dict, list)) else event.get(key) for key in fields})
        return output.getvalue(), "text/csv"
    raise ValueError("Audit format must be 'json' or 'csv'.")
