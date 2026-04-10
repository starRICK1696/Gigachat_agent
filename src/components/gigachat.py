from gigachat import GigaChat
import os
from dataclasses import dataclass
import tempfile
import subprocess
import numpy as np
import logging
from ..lib import prompts

logger = logging.getLogger(__name__)


@dataclass
class GigachatResponse:
    text: str
    tokens: int


def InitGigachatClient(config) -> GigaChat:
    """Initialize GigaChat client using configuration from config.yaml.
    
    Reads certificate_path and api_key from configs/config.yaml.
    """
    gigachat_config = config.get("gigachat", {})
    
    certificate_path = gigachat_config.get("certificate_path", "russian_trusted_root_ca.cer")
    api_key = gigachat_config.get("api_key", "")
    model = gigachat_config.get("model", "GigaChat")
    
    full_path = os.path.abspath(certificate_path)
    
    logger.debug(f"Initializing GigaChat client with certificate: {full_path}")
    logger.debug(f"API key present: {bool(api_key)}")
    logger.debug(f"Using model: {model}")
    
    client = GigaChat(
        credentials=api_key,
        ca_bundle_file=full_path,
        model=model
    )
    logger.info("GigaChat client initialized successfully")
    return client

    
def MakeGigachatRequest(query: str, Client: GigaChat) -> GigachatResponse:
    logger.debug(f"Making GigaChat request with query length: {len(query)} chars")
    logger.debug(f"Request query:\n{query[:500]}{'...' if len(query) > 500 else ''}")
    
    response = Client.chat(query)
    
    response_text = response.choices[0].message.content
    total_tokens = response.usage.total_tokens
    
    logger.debug(f"GigaChat response received - tokens used: {total_tokens}")
    logger.debug(f"Response text:\n{response_text[:500]}{'...' if len(response_text) > 500 else ''}")
    
    return GigachatResponse(response_text, total_tokens)


def CutQueryIfNeeded(last_query: GigachatResponse, Client: GigaChat, max_tokens: int) -> GigachatResponse:
    logger.debug(f"Checking if query needs cutting: {last_query.tokens} tokens vs max {max_tokens}")
    if last_query.tokens < max_tokens:
        logger.debug("Query within token limit, no cutting needed")
        return last_query
    logger.info(f"Query exceeds token limit ({last_query.tokens} > {max_tokens}), requesting context reduction")
    return MakeGigachatRequest(prompts.context_reduction(last_query.text), Client)


# ==================== Legacy single-phase functions (teammate's code) ====================

def MakeClassificationRequest(query: str, Client: GigaChat) -> GigachatResponse:
    logger.info("Making classification request to GigaChat")
    return MakeGigachatRequest(prompts.task_classification(query), Client)


def MakeFormulationRequest(query: str, problem_id: int, input_data: dict, answer: dict, Client: GigaChat) -> GigachatResponse:
    logger.info("Making formulation request to GigaChat")
    return MakeGigachatRequest(prompts.response_formulation(query, problem_id, input_data, answer), Client)


# ==================== Two-phase functions ====================

def ClassifyTaskType(query: str, Client: GigaChat) -> GigachatResponse:
    """Phase 1: Classify the task type from user query.

    The model returns a single number (0-4 or -1 N).

    Args:
        query: User query with conversation context.
        Client: GigaChat client instance.

    Returns:
        GigachatResponse whose ``text`` contains the classification code.
    """
    logger.info("Phase 1: Classifying task type")
    prompt = prompts.get_task_classification_prompt(query)
    return MakeGigachatRequest(prompt, Client)


def ExtractTaskData(query: str, task_id: int, Client: GigaChat) -> GigachatResponse:
    """Phase 2: Extract task-specific data from user query.

    The model returns data in a simple plain-text format (no JSON).

    Args:
        query: User query with conversation context.
        task_id: Task type ID from Phase 1 classification.
        Client: GigaChat client instance.

    Returns:
        GigachatResponse whose ``text`` contains the extracted data.
    """
    logger.info(f"Phase 2: Extracting data for task_id={task_id}")
    prompt = prompts.get_data_extraction_prompt(query, task_id)
    return MakeGigachatRequest(prompt, Client)


def MakeConversationalResponse(query: str, Client: GigaChat) -> GigachatResponse:
    """Generate a regular conversational response (classification code 0).

    Args:
        query: User query with conversation context.
        Client: GigaChat client instance.

    Returns:
        GigachatResponse with a free-text answer.
    """
    logger.info("Generating conversational response")
    prompt = prompts.get_conversational_prompt(query)
    return MakeGigachatRequest(prompt, Client)


def RequestClarification(query: str, task_id: int, Client: GigaChat) -> GigachatResponse:
    """Ask the user for missing data (classification code -1).

    Args:
        query: User query with conversation context.
        task_id: Task type that needs more data.
        Client: GigaChat client instance.

    Returns:
        GigachatResponse with a clarification request.
    """
    logger.info(f"Requesting clarification for task_id={task_id}")
    prompt = prompts.get_clarification_prompt(query, task_id)
    return MakeGigachatRequest(prompt, Client)


def FinalizeTaskResponse(
    query: str, task_name: str, task_result: str, Client: GigaChat
) -> GigachatResponse:
    """Present the solved task result to the user in a friendly way.

    After the solver produces a result, this function asks the model to
    wrap it in a human-readable message.

    Args:
        query: User query with conversation context.
        task_name: Human-readable task name (e.g. "Арифметика").
        task_result: Raw result string from the solver.
        Client: GigaChat client instance.

    Returns:
        GigachatResponse with a nicely formatted answer.
    """
    logger.info(f"Finalizing task response: task_name={task_name}")
    prompt = prompts.get_final_response_prompt(query, task_name, task_result)
    return MakeGigachatRequest(prompt, Client)
