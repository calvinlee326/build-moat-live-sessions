import io
from datetime import datetime

import qrcode
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse, StreamingResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from .database import get_db
from .models import ScanEvent, UrlMapping
from .schemas import CreateRequest, CreateResponse, QRInfoResponse, UpdateRequest
from .token_gen import generate_token
from .url_validator import validate_url

router = APIRouter()

# In-memory dict simulating a Redis cache.
# Key = token string, Value = destination URL.
# Avoids hitting the DB on every QR scan (the hottest path).
redirect_cache: dict[str, str] = {}

BASE_URL = "http://localhost:8000"


@router.post("/api/qr/create", response_model=CreateResponse)
def create_qr(req: CreateRequest, db: Session = Depends(get_db)):
    # Validate and normalize before storing — keeps DB clean
    try:
        normalized_url = validate_url(req.url)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    token = generate_token(normalized_url, db)

    mapping = UrlMapping(
        token=token,
        original_url=normalized_url,
        expires_at=req.expires_at,
    )
    db.add(mapping)
    db.commit()

    short_url = f"{BASE_URL}/r/{token}"

    # Warm the cache on create so the very first scan skips the DB entirely
    redirect_cache[token] = normalized_url

    return CreateResponse(
        token=token,
        short_url=short_url,
        qr_code_url=f"{BASE_URL}/api/qr/{token}/image",
        original_url=normalized_url,
    )


@router.get("/r/{token}")
def redirect(token: str, request: Request, db: Session = Depends(get_db)):
    """Redirect flow: Cache → DB → 410/404.

    This is the hot path — every QR scan hits this endpoint.
    Cache-first keeps DB load low. We still hit the DB on first scan
    after restart or after a cache invalidation (update/delete).
    """
    # Fast path: return from cache without touching the DB
    if token in redirect_cache:
        _record_scan(token, request, db)
        return RedirectResponse(url=redirect_cache[token], status_code=302)

    # Slow path: cache miss, look up in DB
    mapping = db.query(UrlMapping).filter(UrlMapping.token == token).first()
    if mapping is None:
        raise HTTPException(status_code=404, detail="Not Found")

    # 410 Gone (not 404) signals "existed but was removed" — useful for QR
    # codes printed on physical materials where users expect a real page
    if mapping.is_deleted:
        raise HTTPException(status_code=410, detail="Gone")
    if mapping.expires_at and mapping.expires_at < datetime.utcnow():
        raise HTTPException(status_code=410, detail="Gone")

    # Warm cache for future scans, then redirect
    redirect_cache[token] = mapping.original_url
    _record_scan(token, request, db)
    return RedirectResponse(url=mapping.original_url, status_code=302)


@router.get("/api/qr/{token}", response_model=QRInfoResponse)
def get_qr_info(token: str, db: Session = Depends(get_db)):
    mapping = _get_mapping_or_404(token, db)
    return mapping


@router.patch("/api/qr/{token}", response_model=QRInfoResponse)
def update_qr(token: str, req: UpdateRequest, db: Session = Depends(get_db)):
    mapping = _get_mapping_or_404(token, db)

    if req.url is not None:
        try:
            mapping.original_url = validate_url(req.url)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        # Must evict cache so next scan fetches the updated URL from DB
        redirect_cache.pop(token, None)

    if req.expires_at is not None:
        mapping.expires_at = req.expires_at
        # Evict so expiry is rechecked on next scan
        redirect_cache.pop(token, None)

    db.commit()
    # refresh() re-reads the row from DB so the response reflects db defaults/triggers
    db.refresh(mapping)
    return mapping


@router.delete("/api/qr/{token}")
def delete_qr(token: str, db: Session = Depends(get_db)):
    mapping = _get_mapping_or_404(token, db)
    # Soft delete: keep the row so we can return 410 instead of 404
    mapping.is_deleted = True
    db.commit()
    redirect_cache.pop(token, None)  # evict so next scan goes to DB and sees is_deleted
    return {"detail": "Deleted"}


@router.get("/api/qr/{token}/image")
def get_qr_image(token: str, db: Session = Depends(get_db)):
    _get_mapping_or_404(token, db)
    short_url = f"{BASE_URL}/r/{token}"

    # qrcode.make() generates a PIL image of the QR code
    img = qrcode.make(short_url)
    # BytesIO is an in-memory file — avoids writing to disk
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)  # rewind to start so the response reads from the beginning
    return StreamingResponse(buf, media_type="image/png")


@router.get("/api/qr/{token}/analytics")
def get_analytics(token: str, db: Session = Depends(get_db)):
    _get_mapping_or_404(token, db)

    total = db.query(func.count(ScanEvent.id)).filter(ScanEvent.token == token).scalar()

    # GROUP BY date extracts the date part from the timestamp for daily bucketing
    daily = (
        db.query(
            func.date(ScanEvent.scanned_at).label("date"),
            func.count(ScanEvent.id).label("count"),
        )
        .filter(ScanEvent.token == token)
        .group_by(func.date(ScanEvent.scanned_at))
        .all()
    )

    return {
        "token": token,
        "total_scans": total,
        "scans_by_day": [{"date": str(row.date), "count": row.count} for row in daily],
    }


def _get_mapping_or_404(token: str, db: Session) -> UrlMapping:
    # Shared helper so every endpoint uses the same not-found logic
    mapping = db.query(UrlMapping).filter(UrlMapping.token == token).first()
    if mapping is None or mapping.is_deleted:
        raise HTTPException(status_code=404, detail="Not Found")
    return mapping


def _record_scan(token: str, request: Request, db: Session):
    # Write one analytics row per redirect — used by the /analytics endpoint
    event = ScanEvent(
        token=token,
        user_agent=request.headers.get("user-agent"),
        ip_address=request.client.host if request.client else None,
    )
    db.add(event)
    db.commit()
