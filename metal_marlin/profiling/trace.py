"""Chrome trace event helpers for profiling output."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _ns_to_us(value_ns: int) -> int:
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
        self._events: list[TraceEvent] = []
        self._pid = pid
        self._tid = tid

    @property
    def events(self) -> list[TraceEvent]:
        return list(self._events)

    def add_event(self, event: TraceEvent) -> None:
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
        return {
            "traceEvents": [event.to_dict() for event in self._events],
            "displayTimeUnit": "ms",
        }

    def export_json(self, output_path: str | Path) -> None:
        with open(output_path, "w") as handle:
            json.dump(self.to_dict(), handle, indent=2)
