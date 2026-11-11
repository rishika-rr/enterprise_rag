"""Password hashing with scrypt (standard library). Plaintext is never stored (SEC-007)."""
import base64
import hashlib
import hmac
import secrets

_N, _R, _P, _LEN = 2**14, 8, 1, 32


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=_N, r=_R, p=_P, dklen=_LEN)
    return f"scrypt${_N}${_R}${_P}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, expected = stored.split("$")
        if scheme != "scrypt":
            return False
        raw_expected = base64.b64decode(expected)
        digest = hashlib.scrypt(
            password.encode("utf-8"),
            salt=base64.b64decode(salt),
            n=int(n), r=int(r), p=int(p), dklen=len(raw_expected),
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest, raw_expected)


# Verified against when the username is unknown, so that response time does
# not reveal which usernames exist.
DUMMY_HASH = hash_password(secrets.token_urlsafe(16))
