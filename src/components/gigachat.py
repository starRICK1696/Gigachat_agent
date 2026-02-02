from gigachat import GigaChat
import os
from dataclasses import dataclass
import tempfile
import subprocess
import numpy as np
from ..lib import prompts


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
    
    return GigaChat(
        credentials=api_key,
        ca_bundle_file=full_path
    )
    
def MakeGigachatRequest(query: str, Client: GigaChat) -> GigachatResponse:
    response = Client.chat(query)
    return GigachatResponse(response.choices[0].message.content, response.usage.total_tokens)

def CutQueryIfNeeded(last_query: GigachatResponse, Client: GigaChat, max_tokens: int) -> GigachatResponse:
    if last_query.tokens < max_tokens:
        return last_query
    return MakeGigachatRequest(prompts.context_reduction(last_query.text), Client)

def MakeClassificationRequest(query: str, Client: GigaChat) -> GigachatResponse:
    return MakeGigachatRequest(prompts.task_classification(query), Client)
