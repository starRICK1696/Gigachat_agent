import json
import logging
import re
from dataclasses import dataclass
from typing import Optional
from returns.result import Result, Success, Failure

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    result: str
    task_id: Optional[int]


def clean_json_response(text: str) -> str:
    """Remove markdown formatting from JSON response.
    
    Handles cases like:
    - ```json\n{...}\n```
    - ```\n{...}\n```
    - Escaped quotes like \"
    """
    logger.debug(f"Cleaning JSON response: {text[:100]}{'...' if len(text) > 100 else ''}")
    
    cleaned = text.strip()
    
    # Remove markdown code blocks (```json ... ``` or ``` ... ```)
    code_block_pattern = r'^```(?:json)?\s*\n?(.*?)\n?```$'
    match = re.match(code_block_pattern, cleaned, re.DOTALL | re.IGNORECASE)
    if match:
        cleaned = match.group(1).strip()
        logger.debug(f"Removed markdown code block, result: {cleaned[:100]}")
    
    # Handle escaped quotes if the whole thing is wrapped in quotes
    if cleaned.startswith('"') and cleaned.endswith('"'):
        try:
            # Try to parse as a JSON string that contains JSON
            inner = json.loads(cleaned)
            if isinstance(inner, str):
                cleaned = inner
                logger.debug(f"Unwrapped quoted JSON string")
        except json.JSONDecodeError:
            pass
    
    return cleaned


def is_json_response(text: str) -> bool:
    logger.debug(f"Checking if response is JSON: {text[:100]}{'...' if len(text) > 100 else ''}")
    
    # First try raw text
    try:
        json.loads(text)
        logger.debug("Response is valid JSON (raw)")
        return True
    except (json.JSONDecodeError, TypeError):
        pass
    
    # Try after cleaning markdown
    cleaned = clean_json_response(text)
    try:
        json.loads(cleaned)
        logger.debug("Response is valid JSON (after cleaning)")
        return True
    except (json.JSONDecodeError, TypeError) as e:
        logger.debug(f"Response is not JSON: {e}")
        return False


def get_task_summary(task_id: int) -> str:
    task_names: dict[int, str] = {
        1: "TSP",
        2: "Max Clique",
        3: "Knapsack",
        4: "Arithmetic"
    }
    task_name = task_names.get(task_id, "Unknown")
    summary = f"[Task: {task_name}, id={task_id}]"
    logger.debug(f"Task summary: {summary}")
    return summary


def solve_task_from_str(json_response_str: str) -> Result[TaskResult, TaskResult]:
    """
    Принимает JSON-строку, парсит её и решает задачу.
    Автоматически очищает markdown-форматирование.
    """
    logger.info(f"Solving task from JSON string: {json_response_str}")
    
    # Clean markdown formatting before parsing
    cleaned_json = clean_json_response(json_response_str)
    logger.debug(f"Cleaned JSON: {cleaned_json}")
    
    try:
        task_data = json.loads(cleaned_json)
        logger.debug(f"Parsed task data: {task_data}")
        
        task_id = task_data.get("task_id")
        logger.info(f"Task ID: {task_id}")
        
        if task_id == 4:
            expression = task_data.get("data", {}).get("expression")
            logger.debug(f"Arithmetic expression: {expression}")
            
            if not expression:
                logger.warning("No 'expression' field in JSON data")
                return Failure(TaskResult("Ошибка: В JSON нет поля 'expression'", task_id))
            
            try:
                result = eval(expression)
                logger.info(f"Arithmetic result: {expression} = {result}")
                return Success(TaskResult(f'{expression} = {result}', task_id))
            except Exception as e:
                logger.error(f"Eval error for expression '{expression}': {e}")
                return Failure(TaskResult(f"Ошибка вычисления eval: {e}", task_id))

        elif task_id in [1, 2, 3]:
            logger.warning(f"Task type {task_id} is not yet supported")
            return Failure(TaskResult(f"Задача типа {task_id} пока не поддерживается", task_id))
        else:
            logger.warning(f"Unknown task type: {task_id}")
            return Failure(TaskResult("Неизвестный тип задачи", task_id))

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return Failure(TaskResult("Ошибка: На вход подана некорректная JSON-строка", None))
    except Exception as e:
        logger.error(f"System error while solving task: {e}")
        return Failure(TaskResult(f"Системная ошибка: {e}", None))