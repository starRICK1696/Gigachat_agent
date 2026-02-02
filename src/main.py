"""FastAPI application with HTTP endpoint for inserting text into SQLite database."""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from returns.result import Success
import yaml
import logging

from .lib.database import init_db, insert_or_update_context, get_context_by_id
from .lib.task_processing import solve_task_from_str, is_json_response
from .components.models import NewMessageRequest, NewMessageResponse, ErrorResponse
from .components.gigachat import InitGigachatClient, MakeClassificationRequest, CutQueryIfNeeded, GigachatResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    response_model=NewMessageResponse,
    responses={500: {"model": ErrorResponse}},
    summary="Processes user message and updates queury in database",
    description="Processes user message and updates queury in database"
)
async def new_message(request: NewMessageRequest, fastapi_request: Request) -> NewMessageResponse:
    """Processes new message from user.
    
    Args:
        request: Request containing id and user text fields.
        fastapi_request: FastAPI request object to access app state.
        
    Returns:
        Final Gigachat response.
        
    Raises:
        HTTPException: If any operation fails.
    """
    try:
        gigachat_client = fastapi_request.app.state.gigachat_client
        config = fastapi_request.app.state.config
        max_tokens = config.get("gigachat", {}).get("max_tokens", 80000)
        
        context = await get_context_by_id(request.chat_id) or ""
        current_query = f"{context}\nUser: {request.text}" if context else f"User: {request.text}"
        
        for _ in range(config.get("gigachat", {}).get("max_loop_cycles", 3)):
            logger.info("Sending request to GigaChat")
            gigachat_response = MakeClassificationRequest(current_query, gigachat_client)
            response_text = gigachat_response.text
            
            if not is_json_response(response_text):
                logger.info("Received non-JSON response, returning to user")
                await insert_or_update_context(request.chat_id, current_query)
                return NewMessageResponse(gigachat_response=response_text)
            
            logger.info(f"Received JSON response, processing task, {response_text}, \n\n\n current_query: {current_query}\n\n")
            task_result = solve_task_from_str(response_text)
            
            if isinstance(task_result, Success):
                task_data = task_result.unwrap()
            else:
                task_data = task_result.failure()
            
            current_query = f"{current_query}\nAPI response: {task_data.result}"
            
            current_query = CutQueryIfNeeded(
                GigachatResponse(current_query, gigachat_response.tokens),
                gigachat_client,
                max_tokens
            ).text
        logger.error("Error processing request: max loop cycles exceeded")
        raise HTTPException(status_code=500, detail="Error processing request: max loop cycles exceeded")
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}