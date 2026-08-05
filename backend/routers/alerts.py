from fastapi import APIRouter, Depends

from services.alerts import detect_alerts, get_cached_alerts
from services.auth import get_current_user

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


@router.get("/")
async def get_alerts(user=Depends(get_current_user)):
    return get_cached_alerts()


@router.post("/refresh")
async def refresh_alerts(user=Depends(get_current_user)):
    alerts = detect_alerts()
    return {"alerts": alerts, "count": len(alerts)}
