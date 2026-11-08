"""Feature flags: the comma-separated tokens of ZKBOT_FLAGS, as an exact set ("emergency" isn't in "noemergency")."""
from __future__ import annotations

import os

FLAGS: frozenset[str] = frozenset(t.strip() for t in os.environ.get("ZKBOT_FLAGS", "").split(",") if t.strip())
