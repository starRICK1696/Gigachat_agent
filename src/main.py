"""FastAPI application with HTTP endpoint for inserting text into SQLite database."""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from returns.result import Success
import json
import yaml
import logging
import os

from .lib.database import (
    init_db, insert_or_update_context, get_context_by_id, log_gigachat_request,
    create_user, authenticate_user, create_chat, get_user_chats, delete_chat,
    chat_belongs_to_user, update_chat_title, get_messages_json, update_messages_json
)
from .lib.task_processing import (
    solve_task_from_str,
    parse_classification,
    parse_arithmetic_data,
    parse_matrix_data,
    parse_knapsack_data,
    parse_production_scheduling_data,
    parse_multi_knapsack_data,
    parse_tiling_2d_data,
    get_task_summary,
)
from .components.models import (
    NewMessageRequest, NewMessageResponse, ErrorResponse,
    RegisterRequest, LoginRequest, AuthResponse,
    CreateChatRequest, ChatListResponse, ChatInfo, DeleteChatRequest
)
from .components.gigachat import (
    InitGigachatClient, CutQueryIfNeeded, GigachatResponse,
    ClassifyTaskType, ExtractTaskData, MakeConversationalResponse, RequestClarification,
    FinalizeTaskResponse,
)

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


# ==================== Helper to save messages and context ====================

async def _save_response(chat_id: int, user_text: str, assistant_text: str, current_query: str):
    """Persist context and chat messages after producing a final response."""
    await insert_or_update_context(chat_id, current_query)

    existing_messages = json.loads(await get_messages_json(chat_id))
    existing_messages.append({"role": "user", "content": user_text})
    existing_messages.append({"role": "assistant", "content": assistant_text})
    await update_messages_json(chat_id, json.dumps(existing_messages, ensure_ascii=False))


# ==================== Main message endpoint ====================

@app.post(
    "/new_message",
    response_model=NewMessageResponse,
    responses={500: {"model": ErrorResponse}},
    summary="Processes user message and updates queury in database",
    description="Processes user message and updates queury in database"
)
async def new_message(request: NewMessageRequest, fastapi_request: Request) -> NewMessageResponse:
    """Processes new message from user.
    
    The handler uses a **two-phase** approach:

    * **Phase 1** – ask the model to classify the request (returns a number).
    * **Phase 2** – if a task is detected, ask the model to extract the
      task-specific data in a simple plain-text format.

    Args:
        request: Request containing chat_id, user_id and user text fields.
        fastapi_request: FastAPI request object to access app state.
        
    Returns:
        Final Gigachat response.
        
    Raises:
        HTTPException: If any operation fails.
    """
    logger.info("=== NEW MESSAGE REQUEST ===")
    logger.info(f"chat_id: {request.chat_id}, user_id: {request.user_id}")
    logger.info(f"text: {request.text}")
    
    # Verify chat belongs to user
    if not await chat_belongs_to_user(request.chat_id, request.user_id):
        raise HTTPException(status_code=403, detail="Chat does not belong to this user")
    
    current_query = f"User: {request.text}"  # Initialize early for error logging
    
    try:
        gigachat_client = fastapi_request.app.state.gigachat_client
        config = fastapi_request.app.state.config
        max_tokens = config.get("gigachat", {}).get("max_tokens", 80000)
        max_loop_cycles = config.get("gigachat", {}).get("max_loop_cycles", 3)
        
        logger.debug(f"Config: max_tokens={max_tokens}, max_loop_cycles={max_loop_cycles}")
        
        context = await get_context_by_id(request.chat_id) or ""
        logger.debug(f"Retrieved context for chat_id={request.chat_id}: {len(context)} chars")
        
        # Update chat title with first message text
        if not context:
            title = request.text[:40] + ("..." if len(request.text) > 40 else "")
            await update_chat_title(request.chat_id, title)
        
        current_query = f"{context}\nUser: {request.text}" if context else f"User: {request.text}"
        logger.debug(f"Initial query constructed, length: {len(current_query)} chars")
        
        for cycle in range(max_loop_cycles):
            logger.info(f"--- Loop cycle {cycle + 1}/{max_loop_cycles} ---")
            logger.debug(f"Current query:\n{current_query[:1000]}{'...' if len(current_query) > 1000 else ''}")
            
            # ========== PHASE 1: TASK CLASSIFICATION ==========
            logger.info("=== PHASE 1: Task Classification ===")
            classification_response = ClassifyTaskType(current_query, gigachat_client)
            
            logger.info(
                f"Classification response: '{classification_response.text}', "
                f"tokens: {classification_response.tokens}"
            )
            
            # Parse the classification code
            try:
                classification_code, clarification_task_id = parse_classification(
                    classification_response.text
                )
            except Exception as e:
                logger.error(f"Failed to parse classification: {e}")
                classification_code = 0  # Default to conversation
                clarification_task_id = None
            
            logger.info(
                f"Classification code: {classification_code}, "
                f"clarification_task_id: {clarification_task_id}"
            )
            
            # ========== CODE 0: CONVERSATION ==========
            if classification_code == 0:
                logger.info("Classification: CONVERSATION — generating text response")
                
                conversational_response = MakeConversationalResponse(
                    current_query, gigachat_client
                )
                
                await log_gigachat_request(
                    chat_id=request.chat_id,
                    request_text=current_query,
                    response_text=conversational_response.text,
                    tokens_used=classification_response.tokens + conversational_response.tokens,
                    is_json_response=False,
                    task_id=None,
                    task_result=None,
                    error=None,
                )
                
                await _save_response(
                    request.chat_id, request.text,
                    conversational_response.text, current_query,
                )
                
                logger.info("=== REQUEST COMPLETED (conversation) ===")
                return NewMessageResponse(gigachat_response=conversational_response.text)
            
            # ========== CODE -1: CLARIFICATION NEEDED ==========
            if classification_code == -1:
                logger.info(
                    f"Classification: CLARIFICATION NEEDED for task_id={clarification_task_id}"
                )
                
                clarification_response = RequestClarification(
                    current_query, clarification_task_id or 0, gigachat_client
                )
                
                await log_gigachat_request(
                    chat_id=request.chat_id,
                    request_text=current_query,
                    response_text=clarification_response.text,
                    tokens_used=classification_response.tokens + clarification_response.tokens,
                    is_json_response=False,
                    task_id=clarification_task_id,
                    task_result="Clarification requested",
                    error=None,
                )
                
                await _save_response(
                    request.chat_id, request.text,
                    clarification_response.text, current_query,
                )
                
                logger.info("=== REQUEST COMPLETED (clarification) ===")
                return NewMessageResponse(gigachat_response=clarification_response.text)
            
            # ========== CODES 1-8: TASK ==========
            if classification_code in [1, 2, 3, 4, 5, 6, 7, 8]:
                logger.info(f"Classification: TASK (task_id={classification_code})")
                
                # ========== PHASE 2: DATA EXTRACTION ==========
                logger.info("=== PHASE 2: Data Extraction ===")
                extraction_response = ExtractTaskData(
                    current_query, classification_code, gigachat_client
                )
                
                logger.debug(f"Extraction response: {extraction_response.text}")
                
                # Parse extracted data depending on task type and build JSON
                # that solve_task_from_str() expects.
                task_json = ""
                try:
                    if classification_code == 4:  # Arithmetic
                        expression = parse_arithmetic_data(extraction_response.text)
                        task_json = json.dumps({
                            "task_id": 4,
                            "data": {"expression": expression}
                        })
                        logger.info(f"Arithmetic expression: {expression}")
                        
                    elif classification_code == 1:  # TSP
                        distances = parse_matrix_data(extraction_response.text)
                        task_json = json.dumps({
                            "task_id": 1,
                            "data": {"distances": distances}
                        })
                        logger.info(f"TSP distances matrix: {distances}")
                        
                    elif classification_code == 2:  # Max Clique
                        adjacency = parse_matrix_data(extraction_response.text)
                        task_json = json.dumps({
                            "task_id": 2,
                            "data": {"adjacency_matrix": adjacency}
                        })
                        logger.info(f"Max Clique adjacency matrix: {adjacency}")
                        
                    elif classification_code == 3:  # Knapsack
                        knapsack_data = parse_knapsack_data(extraction_response.text)
                        task_json = json.dumps({
                            "task_id": 3,
                            "data": knapsack_data
                        })
                        logger.info(f"Knapsack data: {knapsack_data}")
                        
                    elif classification_code == 5:  # Max Weight Clique
                        adjacency = parse_matrix_data(extraction_response.text)
                        task_json = json.dumps({
                            "task_id": 5,
                            "data": {"adjacency_matrix": adjacency}
                        })
                        logger.info(f"Max Weight Clique adjacency matrix: {adjacency}")
                        
                    elif classification_code == 6:  # Production Scheduling
                        scheduling_data = parse_production_scheduling_data(extraction_response.text)
                        task_json = json.dumps({
                            "task_id": 6,
                            "data": scheduling_data
                        })
                        logger.info(f"Production Scheduling data: {scheduling_data}")

                    elif classification_code == 7:  # Multi-Knapsack
                        multi_knapsack_data = parse_multi_knapsack_data(extraction_response.text)
                        task_json = json.dumps({
                            "task_id": 7,
                            "data": multi_knapsack_data
                        })
                        logger.info(f"Multi-Knapsack data: {multi_knapsack_data}")

                    elif classification_code == 8:  # 2D Tiling
                        tiling_data = parse_tiling_2d_data(extraction_response.text)
                        task_json = json.dumps({
                            "task_id": 8,
                            "data": tiling_data
                        })
                        logger.info(f"2D Tiling data: {tiling_data}")
                    
                except Exception as e:
                    logger.error(f"Failed to parse task data: {e}")
                    
                    await log_gigachat_request(
                        chat_id=request.chat_id,
                        request_text=current_query,
                        response_text=extraction_response.text,
                        tokens_used=(
                            classification_response.tokens + extraction_response.tokens
                        ),
                        is_json_response=False,
                        task_id=classification_code,
                        task_result=None,
                        error=f"Data parsing failed: {e}",
                    )
                    
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to parse task data: {e}",
                    )
                
                # ========== SOLVE THE TASK ==========
                logger.info("=== TASK PROCESSING ===")
                task_result = solve_task_from_str(task_json)
                
                if isinstance(task_result, Success):
                    task_data = task_result.unwrap()
                    logger.info(f"Task solved successfully: {task_data.result}")
                else:
                    task_data = task_result.failure()
                    logger.warning(f"Task failed: {task_data.result}")
                
                total_tokens = (
                    classification_response.tokens + extraction_response.tokens
                )
                
                # ========== FINALIZE: Present result to user ==========
                task_name = get_task_summary(classification_code)
                logger.info(f"Generating final response for task: {task_name}")
                
                final_response = FinalizeTaskResponse(
                    current_query, task_name, task_data.result, gigachat_client
                )
                total_tokens += final_response.tokens
                
                await log_gigachat_request(
                    chat_id=request.chat_id,
                    request_text=current_query,
                    response_text=final_response.text,
                    tokens_used=total_tokens,
                    is_json_response=True,
                    task_id=task_data.task_id,
                    task_result=task_data.result,
                    error=None if isinstance(task_result, Success) else "Task processing failed",
                )
                
                await _save_response(
                    request.chat_id, request.text,
                    final_response.text, current_query,
                )
                
                logger.info("=== REQUEST COMPLETED (task solved) ===")
                return NewMessageResponse(gigachat_response=final_response.text)
            
            # Unknown classification code
            logger.warning(f"Unknown classification code: {classification_code}")
            raise HTTPException(
                status_code=500,
                detail=f"Unknown classification code: {classification_code}",
            )
            
        # Exited the loop — max cycles exceeded
        logger.error(f"Max loop cycles ({max_loop_cycles}) exceeded without final response")
        logger.info("=== REQUEST FAILED (max cycles exceeded) ===")
        
        await log_gigachat_request(
            chat_id=request.chat_id,
            request_text=current_query,
            response_text=None,
            tokens_used=None,
            is_json_response=None,
            task_id=None,
            task_result=None,
            error="Max loop cycles exceeded",
        )
        
        raise HTTPException(
            status_code=500,
            detail="Error processing request: max loop cycles exceeded",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing request: {e}", exc_info=True)
        logger.info("=== REQUEST FAILED (exception) ===")
        
        await log_gigachat_request(
            chat_id=request.chat_id,
            request_text=current_query,
            response_text=None,
            tokens_used=None,
            is_json_response=None,
            task_id=None,
            task_result=None,
            error=str(e),
        )
        
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


# ==================== Auth endpoints ====================

@app.post(
    "/register",
    response_model=AuthResponse,
    responses={409: {"model": ErrorResponse}},
    summary="Register a new user",
    description="Create a new user account with username and password"
)
async def register(request: RegisterRequest) -> AuthResponse:
    """Register a new user."""
    user_id = await create_user(request.username, request.password)
    if user_id is None:
        raise HTTPException(status_code=409, detail="Username already exists")
    return AuthResponse(user_id=user_id, username=request.username)


@app.post(
    "/login",
    response_model=AuthResponse,
    responses={401: {"model": ErrorResponse}},
    summary="Login",
    description="Authenticate with username and password"
)
async def login(request: LoginRequest) -> AuthResponse:
    """Authenticate a user."""
    user_id = await authenticate_user(request.username, request.password)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return AuthResponse(user_id=user_id, username=request.username)


# ==================== Chat management endpoints ====================

@app.post(
    "/chats",
    response_model=ChatInfo,
    summary="Create a new chat",
    description="Create a new chat for the authenticated user"
)
async def create_new_chat(request: CreateChatRequest) -> ChatInfo:
    """Create a new chat."""
    chat_id = await create_chat(request.user_id, request.title)
    return ChatInfo(id=chat_id, title=request.title, created_at="")


@app.get(
    "/chats/{user_id}",
    response_model=ChatListResponse,
    summary="Get user chats",
    description="Get all chats for a user"
)
async def get_chats(user_id: int) -> ChatListResponse:
    """Get all chats for a user."""
    chats = await get_user_chats(user_id)
    return ChatListResponse(
        chats=[ChatInfo(id=c["id"], title=c["title"], created_at=c["created_at"]) for c in chats]
    )


@app.delete(
    "/chats",
    summary="Delete a chat",
    description="Delete a chat belonging to the user"
)
async def remove_chat(request: DeleteChatRequest):
    """Delete a chat."""
    deleted = await delete_chat(request.chat_id, request.user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Chat not found or does not belong to user")
    return {"status": "success", "message": "Chat deleted"}


@app.get(
    "/chats/{chat_id}/messages",
    summary="Get chat messages",
    description="Get all messages for a specific chat"
)
async def get_chat_messages(chat_id: int, user_id: int):
    """Get messages for a chat."""
    if not await chat_belongs_to_user(chat_id, user_id):
        raise HTTPException(status_code=403, detail="Chat does not belong to this user")
    
    messages_json = await get_messages_json(chat_id)
    return {"messages": json.loads(messages_json)}
