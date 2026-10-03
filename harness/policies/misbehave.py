"""Deliberately misbehaving policies, used only by the protocol tests: each
one must make the Go simulator abort the run with a clear error."""
from __future__ import annotations

import os
import sys
import time


class Sleep:
    """Never answers in time (sleeps far beyond any test timeout)."""

    name = "test:sleep"

    def __init__(self, params, seed, cluster):
        pass

    def schedule(self, view):
        time.sleep(3600)
        return {"actions": []}


class BadJSON:
    """Writes a line that is not JSON."""

    name = "test:badjson"
    raw_reply = b"this is not json\n"

    def __init__(self, params, seed, cluster):
        pass

    def schedule(self, view):
        return None  # the server writes raw_reply instead


class Invalid:
    """Starts a job that does not exist."""

    name = "test:invalid"

    def __init__(self, params, seed, cluster):
        pass

    def schedule(self, view):
        return {"actions": [{"op": "start", "job_id": "no-such-job", "placement": [{"node": view.nodes[0].name, "workers": 1}]}]}


class Crash:
    """Exits in the middle of the session."""

    name = "test:crash"

    def __init__(self, params, seed, cluster):
        pass

    def schedule(self, view):
        sys.stderr.write("test:crash exits now\n")
        sys.stderr.flush()
        os._exit(3)
