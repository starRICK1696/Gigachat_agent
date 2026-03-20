"""FastAPI application with HTTP endpoint for inserting text into SQLite database."""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from returns.result import Success
import yaml
import logging
import os

from .lib.database import init_db, insert_or_update_context, get_context_by_id, log_gigachat_request
from .lib.task_processing import solve_task_from_str, is_json_response
from .components.models import NewMessageRequest, NewMessageResponse, ErrorResponse
from .components.gigachat import InitGigachatClient, MakeClassificationRequest, CutQueryIfNeeded, GigachatResponse

# Configure logging level from environment variable, default to INFO
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
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

# Add CORS middleware to allow requests from the HTML page
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
async def root():
    """Serve the main HTML page."""
    logger.debug("Serving index.html")
    return FileResponse("static/index.html")


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
    logger.info(f"=== NEW MESSAGE REQUEST ===")
    logger.info(f"chat_id: {request.chat_id}")
    logger.info(f"text: {request.text}")
    
    current_query = f"User: {request.text}"  # Initialize early for error logging
    
    try:
        gigachat_client = fastapi_request.app.state.gigachat_client
        config = fastapi_request.app.state.config
        max_tokens = config.get("gigachat", {}).get("max_tokens", 80000)
        max_loop_cycles = config.get("gigachat", {}).get("max_loop_cycles", 3)
        
        logger.debug(f"Config: max_tokens={max_tokens}, max_loop_cycles={max_loop_cycles}")
        
        context = await get_context_by_id(request.chat_id) or ""
        logger.debug(f"Retrieved context for chat_id={request.chat_id}: {len(context)} chars")
        
        current_query = f"{context}\nUser: {request.text}" if context else f"User: {request.text}"
        logger.debug(f"Initial query constructed, length: {len(current_query)} chars")
        
        for cycle in range(max_loop_cycles):
            logger.info(f"--- Loop cycle {cycle + 1}/{max_loop_cycles} ---")
            logger.debug(f"Current query:\n{current_query[:1000]}{'...' if len(current_query) > 1000 else ''}")
            
            logger.info("Sending classification request to GigaChat")
            gigachat_response = MakeClassificationRequest(current_query, gigachat_client)
            response_text = gigachat_response.text
            
            logger.info(f"GigaChat response received, length: {len(response_text)} chars, tokens: {gigachat_response.tokens}")
            logger.debug(f"Response text:\n{response_text}")
            
            is_json = is_json_response(response_text)
            
            if not is_json:
                logger.info("Response is not JSON - returning to user as final answer")
                
                # Log to database
                await log_gigachat_request(
                    chat_id=request.chat_id,
                    request_text=current_query,
                    response_text=response_text,
                    tokens_used=gigachat_response.tokens,
                    is_json_response=False,
                    task_id=None,
                    task_result=None,
                    error=None
                )
                
                await insert_or_update_context(request.chat_id, current_query)
                logger.info(f"=== REQUEST COMPLETED (non-JSON response) ===")
                return NewMessageResponse(gigachat_response=response_text)
            
            logger.info(f"Response is JSON - processing as task")
            logger.debug(f"JSON response: {response_text}")
            
            task_result = solve_task_from_str(response_text)
            
            if isinstance(task_result, Success):
                task_data = task_result.unwrap()
                logger.info(f"Task solved successfully: {task_data.result}")
            else:
                task_data = task_result.failure()
                logger.warning(f"Task failed: {task_data.result}")
            
            # Log to database
            await log_gigachat_request(
                chat_id=request.chat_id,
                request_text=current_query,
                response_text=response_text,
                tokens_used=gigachat_response.tokens,
                is_json_response=True,
                task_id=task_data.task_id,
                task_result=task_data.result,
                error=None if isinstance(task_result, Success) else "Task processing failed"
            )
            
            current_query = f"{current_query}\nAPI response: {task_data.result}"
            logger.debug(f"Updated query with API response, new length: {len(current_query)} chars")
            
            logger.debug("Checking if query needs to be cut due to token limit")
            current_query = CutQueryIfNeeded(
                GigachatResponse(current_query, gigachat_response.tokens),
                gigachat_client,
                max_tokens
            ).text
            logger.debug(f"Query after potential cutting: {len(current_query)} chars")
            
        logger.error(f"Max loop cycles ({max_loop_cycles}) exceeded without final response")
        logger.info(f"=== REQUEST FAILED (max cycles exceeded) ===")
        
        # Log error to database
        await log_gigachat_request(
            chat_id=request.chat_id,
            request_text=current_query,
            response_text=None,
            tokens_used=None,
            is_json_response=None,
            task_id=None,
            task_result=None,
            error="Max loop cycles exceeded"
        )
        
        raise HTTPException(status_code=500, detail="Error processing request: max loop cycles exceeded")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing request: {e}", exc_info=True)
        logger.info(f"=== REQUEST FAILED (exception) ===")
        
        # Log error to database
        await log_gigachat_request(
            chat_id=request.chat_id,
            request_text=current_query,
            response_text=None,
            tokens_used=None,
            is_json_response=None,
            task_id=None,
            task_result=None,
            error=str(e)
        )
        
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}