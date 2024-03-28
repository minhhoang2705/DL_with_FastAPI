from fastapi import APIRouter
from .catdog_route import router as catdog_router

router = APIRouter()
router.include_router(catdog_router, prefix="/catdog_classification")