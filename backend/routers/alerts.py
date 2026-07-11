from fastapi import APIRouter
from services.alerts import get_cached_alerts, detect_alerts

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


@router.get("/")
async def get_alerts():
    return get_cached_alerts()


@router.post("/refresh")
async def refresh_alerts():
    alerts = detect_alerts()
    return {"alerts": alerts, "count": len(alerts)}