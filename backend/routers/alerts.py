import asyncio

from fastapi import APIRouter, Depends

from services.alerts import detect_alerts, get_cached_alerts
from services.auth import get_current_user

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


@router.get("/")
async def get_alerts(user=Depends(get_current_user)):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_cached_alerts)


@router.post("/refresh")
async def refresh_alerts(user=Depends(get_current_user)):
    loop = asyncio.get_event_loop()
    alerts = await loop.run_in_executor(None, detect_alerts)
    return {"alerts": alerts, "count": len(alerts)}
