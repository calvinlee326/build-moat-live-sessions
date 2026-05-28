import hashlib
import string
import time

from sqlalchemy.orm import Session

from .models import UrlMapping

# Base62 uses letters + digits (no special chars) so tokens are URL-safe
BASE62_CHARS = string.ascii_letters + string.digits  # a-zA-Z0-9
TOKEN_LENGTH = 7   # 62^7 ≈ 3.5 trillion possible tokens
MAX_RETRIES = 10   # retry on hash collision before giving up


def base62_encode(data: bytes) -> str:
    """Convert bytes to Base62 string.

    Why Base62? URL-safe (no +/= like Base64), short, and human-readable.
    This is the same idea behind bit.ly and YouTube video IDs.
    """
    # Treat the raw bytes as one big integer (big-endian)
    num = int.from_bytes(data, "big")
    if num == 0:
        return BASE62_CHARS[0]
    result = []
    # Repeatedly divide by 62, each remainder maps to one Base62 character
    while num > 0:
        num, remainder = divmod(num, 62)
        result.append(BASE62_CHARS[remainder])
    # divmod builds digits least-significant first, so reverse at the end
    return "".join(reversed(result))


def token_exists_in_db(db: Session, token: str) -> bool:
    # Returns True if this token is already taken — used for collision detection
    return db.query(UrlMapping).filter(UrlMapping.token == token).first() is not None


def generate_token(url: str, db: Session) -> str:
    """SHA-256 + nonce + Base62 token generation with collision retry.

    Why not random UUIDs? SHA-256 gives us deterministic-ish short IDs.
    Why add a nonce? Two different requests for the same URL would produce
    the same hash without one — but we want unique tokens per request.
    Why retry? Collision probability is low but non-zero; as the table
    fills up, short prefixes of the hash start to repeat.
    """
    for attempt in range(MAX_RETRIES):
        # Mix in attempt index + nanoseconds so each retry produces a different hash
        nonce = f"{attempt}:{time.time_ns()}"
        digest = hashlib.sha256(f"{url}{nonce}".encode()).digest()
        # Take only the first TOKEN_LENGTH characters of the Base62 string
        token = base62_encode(digest)[:TOKEN_LENGTH]
        if not token_exists_in_db(db, token):
            return token
    raise RuntimeError("Failed to generate a unique token after max retries")
