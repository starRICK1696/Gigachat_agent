"""FastAPI application with HTTP endpoint for inserting text into SQLite database."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

import yaml
from lib.database import init_db, insert_or_update_text
from components.models import InsertRequest, InsertResponse, ErrorResponse
from components.gigachat import InitGigachatClient


CONFIG_PATH = "configs/config.yaml"


def load_config() -> dict:
    """Load configuration from YAML file."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database and GigaChat client on startup."""
    await init_db()
    config = load_config()
    app.state.config = config

    gigachat_client = InitGigachatClient(config)
    app.state.gigachat_client = gigachat_client
    
    yield

    gigachat_client.close()


app = FastAPI(
    title="Gigachat agent API",
    description="API for agent of gigachet for solving NP complite tasks",
    version="1.0.0",
    lifespan=lifespan
)


@app.post(
    "/new_message",
    response_model=InsertResponse,
    responses={500: {"model": ErrorResponse}},
    summary="Processes user message and updates queury in database",
    description="Processes user message and updates queury in database"
)
async def insert_text(request: InsertRequest, fastapi_request: Request) -> InsertResponse:
    """Insert or update text in the database.
    
    Args:
        request: Request containing id and user text fields.
        fastapi_request: FastAPI request object to access app state.
        
    Returns:
        Final Gigachat response.
        
    Raises:
        HTTPException: If any operation fails.
    
    Example usage of GigaChat client:
        # Access the GigaChat client from app state
        gigachat_client = fastapi_request.app.state.gigachat_client
        
        # Use the client with helper functions from components.gigachat:
        # from components.gigachat import MakeGigachatRequest, MakeClassificationRequest
        # response = MakeGigachatRequest("Your query here", gigachat_client)
        # classification = MakeClassificationRequest("Your query", gigachat_client)
    """
    try:
        # Example: Access GigaChat client (uncomment to use)
        # gigachat_client = fastapi_request.app.state.gigachat_client
        # response = MakeGigachatRequest(request.text, gigachat_client)
        
        await insert_or_update_text(request.id, request.text)
        return InsertResponse(id=request.id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}