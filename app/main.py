from fastapi import FastAPI
from app.api.v1.endpoints import router as api_v1_router
from app.core.config import settings
from app.db.session import engine, Base
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Tables are now managed by Alembic. 
    # For local dev/testing without manual migrations, you could keep this,
    # but for production it should be handled via migration scripts.
    yield

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Service is healthy"}

app.include_router(api_v1_router, prefix=settings.API_V1_STR)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
