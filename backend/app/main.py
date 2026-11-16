from pydantic import BaseModel, Field
import json
from app.api.marketplace import router as marketplace_router
from app.agents.state import create_initial_state
from app.graph.workflow import workflow
from fastapi import FastAPI, HTTPException
from app.api.customers import router as customers_router
from app.api.products import router as products_router
from app.api.sellers import router as sellers_router
from app.api.reviews import router as reviews_router
from app.api.payments import router as payments_router
from app.database.connection import check_database_connection

app = FastAPI(
    title="InsightOps AI",
    description="Autonomous Business Intelligence Platform",
    version="0.1.0"
)


class AnalysisRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="Business question to analyze"
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


@app.post("/api/analyze")
def analyze(request: AnalysisRequest):

    try:
        initial_state = create_initial_state(request.query)

        final_state = workflow.invoke(initial_state)

        # Extract generated report
        report = final_state.get("final_answer", "")

        # Convert JSON string into Python dictionary
        if isinstance(report, str):
            try:
                report = json.loads(report)
            except json.JSONDecodeError:
                pass

        return {
            "query": request.query,
            "execution_status": final_state.get("execution_status"),
            "validation_status": final_state.get("validation_status"),
            "report": report,
            "completed_agents": final_state.get("completed_agents", []),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Analysis workflow execution failed"
        ) from exc

# MarketPlace API routes
app.include_router(marketplace_router)

app.include_router(customers_router)

app.include_router(products_router)

app.include_router(sellers_router)

app.include_router(reviews_router)

app.include_router(payments_router)