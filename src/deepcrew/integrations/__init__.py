"""
Optional third-party framework integrations.

Each integration is installed via its own extra (e.g. ``pip install
deepcrew-ai[fastapi]``) and imports its dependency lazily, so a bare
``pip install deepcrew-ai`` never pulls in fastapi or any other optional
framework.
"""

from __future__ import annotations
