"""Chrome trace event helpers for profiling output."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any



logger = logging.getLogger(__name__)

def _ns_to_us(value_ns: int) -> int:
    logger.debug("_ns_to_us called with value_ns=%s", value_ns)
    return int(value_ns / 1000)


@dataclass(frozen=True)
class TraceEvent:
    """Single Chrome trace event."""

    name: str
    cat: str
    ph: str
    ts: int
    dur: int = 0
    pid: int = 0
    tid: int = 0
    args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        logger.debug("to_dict called")
        event = {
            "name": self.name,
            "cat": self.cat,
            "ph": self.ph,
            "ts": self.ts,
            "pid": self.pid,
            "tid": self.tid,
            "args": self.args,
        }
        if self.ph == "X":
            event["dur"] = self.dur
        return event


class ChromeTrace:
    """Chrome trace event container."""

    def __init__(self, *, pid: int = 0, tid: int = 0):
        logger.debug("initializing %s", type(self).__name__)
        self._events: list[TraceEvent] = []
        self._pid = pid
        self._tid = tid

    @property
    def events(self) -> list[TraceEvent]:
        logger.debug("events called")
        return list(self._events)

    def add_event(self, event: TraceEvent) -> None:
        logger.debug("add_event called with event=%s", event)
        self._events.append(event)

    def add_duration(
        self,
        *,
        name: str,
        start_ns: int,
        end_ns: int,
        cat: str = "kernel",
        args: dict[str, Any] | None = None,
        pid: int | None = None,
        tid: int | None = None,
    ) -> None:
        logger.debug("add_duration called")
        self._events.append(
            TraceEvent(
                name=name,
                cat=cat,
                ph="X",
                ts=_ns_to_us(start_ns),
                dur=_ns_to_us(end_ns - start_ns),
                pid=self._pid if pid is None else pid,
                tid=self._tid if tid is None else tid,
                args=args or {},
            )
        )

    def add_counter(
        self,
        *,
        name: str,
        timestamp_ns: int,
        cat: str = "metrics",
        args: dict[str, Any],
        pid: int | None = None,
        tid: int | None = None,
    ) -> None:
        logger.debug("add_counter called")
        self._events.append(
            TraceEvent(
                name=name,
                cat=cat,
                ph="C",
                ts=_ns_to_us(timestamp_ns),
                pid=self._pid if pid is None else pid,
                tid=self._tid if tid is None else tid,
                args=args,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        logger.debug("to_dict called")
        return {
            "traceEvents": [event.to_dict() for event in self._events],
            "displayTimeUnit": "ms",
        }

    def export_json(self, output_path: str | Path) -> None:
        logger.info("export_json called with output_path=%s", output_path)
        with open(output_path, "w") as handle:
            json.dump(self.to_dict(), handle, indent=2)
