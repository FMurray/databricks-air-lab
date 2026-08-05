"""Who is using the app, and which teams they belong to.

Resolution order:
  1. Databricks Apps: the platform injects the authenticated principal via the
     X-Forwarded-Email/X-Forwarded-User headers (never trust a user-typed identity).
  2. Local development: HUB_USER env var, else the OS user (clearly labeled as dev identity).

Team membership comes from config — users never pick a team; zero memberships means
read-only (that is the access gate).
"""

from __future__ import annotations

import getpass
import os


def current_user() -> tuple[str, str]:
    """Returns (principal, source)."""
    try:
        import streamlit as st
        headers = getattr(st.context, "headers", {}) or {}
        for h in ("X-Forwarded-Email", "X-Forwarded-User"):
            if headers.get(h):
                return headers[h], "databricks-apps"
    except Exception:
        pass
    if os.environ.get("HUB_USER"):
        return os.environ["HUB_USER"], "env"
    return f"{getpass.getuser()} (local dev)", "os"
