from urllib.parse import urlparse

MAX_URL_LENGTH = 2048  # browsers and servers commonly cap URLs here

# In production this would be a real blocklist (e.g. Google Safe Browsing API)
BLOCKED_DOMAINS = {
    "evil.com",
    "malware.example.com",
    "phishing.example.com",
}


def is_blocked_domain(hostname: str | None) -> bool:
    if hostname is None:
        return True
    return hostname.lower() in BLOCKED_DOMAINS


def validate_url(url: str) -> str:
    """Format check, normalization, and blocklist validation.

    Normalization matters because http://Example.com/ and https://example.com
    are the same destination — without normalizing, they'd get different tokens
    and waste space in the DB.
    """
    if len(url) > MAX_URL_LENGTH:
        raise ValueError(f"URL exceeds maximum length of {MAX_URL_LENGTH}")

    # urlparse splits a URL into its components (scheme, host, path, etc.)
    # e.g. urlparse("https://example.com/path") -> scheme="https", netloc="example.com", ...
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("URL must use http or https scheme")
    if is_blocked_domain(parsed.hostname):
        raise ValueError(f"Domain '{parsed.hostname}' is blocked")

    # _replace() returns a new ParseResult with only the specified fields changed.
    # We lowercase the host and upgrade http → https for consistent storage.
    normalized = parsed._replace(
        scheme="https",
        netloc=parsed.netloc.lower(),
    ).geturl().rstrip("/")  # strip trailing slash so /path/ and /path map to the same token
    return normalized
