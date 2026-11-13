
from fastapi import FastAPI, HTTPException

from app.database.connection import check_database_connection

app = FastAPI(
    title="InsightOps AI",
    description="Autonomous Business Intelligence Platform",
    version="0.1.0"
)


@app.get("/")
def root():
    return {
        "message": "Welcome to InsightOps AI",
        "status": "running"
    }


@app.get("/health")
def health():
    try:
        db_status = check_database_connection()

        return {
            "api": "healthy",
            "database": "connected",
            "database_test": db_status
        }

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Database connection failed"
        ) from exc
