"""
API-key authentication and role-based access to document collections.

Configure keys with the API_KEYS env var:

    API_KEYS="k-eng-123:engineering,k-crew-456:cabin_crew|flight_ops,k-admin-789:admin"

Each key maps to one or more roles (separated by |). When API_KEYS is not set the API
runs in *development mode*: every request is treated as an admin and a warning is logged.
In an enterprise deployment this module is where SSO / OAuth2 (e.g. Entra ID JWTs) plugs in.
"""

import hmac
import os
from dataclasses import dataclass, field

from logger.custom_logger import CustomLogger

log = CustomLogger().get_logger(__name__)
ADMIN_ROLE = "admin"


@dataclass(frozen=True)
class Principal:
    name: str
    roles: frozenset = field(default_factory=frozenset)

    @property
    def is_admin(self) -> bool:
        return ADMIN_ROLE in self.roles


DEV_PRINCIPAL = Principal(name="dev-mode", roles=frozenset({ADMIN_ROLE}))


def parse_api_keys(raw: str | None) -> dict[str, frozenset]:
    keys: dict[str, frozenset] = {}
    for entry in (raw or "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        key, _, roles = entry.partition(":")
        role_set = frozenset(r.strip().lower() for r in roles.split("|") if r.strip())
        if key.strip() and role_set:
            keys[key.strip()] = role_set
    return keys


def auth_enabled() -> bool:
    return bool(parse_api_keys(os.getenv("API_KEYS")))


def authenticate(api_key: str | None) -> Principal | None:
    """Return the caller's Principal, DEV_PRINCIPAL when auth is disabled, or None if the key is invalid."""
    keys = parse_api_keys(os.getenv("API_KEYS"))
    if not keys:
        log.warning("API_KEYS not set - running in development mode without authentication")
        return DEV_PRINCIPAL
    if not api_key:
        return None
    for known_key, roles in keys.items():
        if hmac.compare_digest(known_key.encode(), api_key.encode()):  # constant-time comparison
            return Principal(name=f"key-{known_key[-4:]}", roles=roles)
    return None


def can_access(principal: Principal, allowed_roles) -> bool:
    """Admins see everything; otherwise the caller needs one of the collection's roles (empty = everyone)."""
    allowed = {r.lower() for r in (allowed_roles or [])}
    return principal.is_admin or not allowed or bool(allowed & principal.roles)
