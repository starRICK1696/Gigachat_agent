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
    
    full_path = os.path.abspath(certificate_path)
    
    logger.debug(f"Initializing GigaChat client with certificate: {full_path}")
    logger.debug(f"API key present: {bool(api_key)}")
    
    client = GigaChat(
        credentials=api_key,
        ca_bundle_file=full_path
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

def MakeClassificationRequest(query: str, Client: GigaChat) -> GigachatResponse:
    logger.info("Making classification request to GigaChat")
    return MakeGigachatRequest(prompts.task_classification(query), Client)
