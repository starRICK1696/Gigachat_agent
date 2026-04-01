import json
import logging
import re
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from returns.result import Result, Success, Failure
import numpy as np

from .compute import SalesmanEvaluator, CliqueEvaluator, KnapsackEvaluator

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    result: str
    task_id: Optional[int]


# ==================== Phase 1: Classification Parsing ====================

def parse_classification(response_text: str) -> Tuple[int, Optional[int]]:
    """Parse Phase 1 classification response.

    The model returns a single number (0-4) or ``-1 N`` when clarification
    is needed for task type *N*.

    Args:
        response_text: Raw model response (should contain a number).

    Returns:
        ``(classification_code, clarification_task_id)``

    Examples:
        ``"4"``      → ``(4, None)``
        ``"-1 3"``   → ``(-1, 3)``
        ``"0"``      → ``(0, None)``
        ``"Ответ: 4"`` → ``(4, None)``
    """
    logger.debug(f"Parsing classification: {response_text}")

    text = response_text.strip()

    # Extract all integers (including negative) from the response
    numbers = re.findall(r'-?\d+', text)

    if not numbers:
        logger.warning(
            "No numbers found in classification response, defaulting to 0 (conversation)"
        )
        return (0, None)

    classification = int(numbers[0])
    logger.debug(f"Classification code: {classification}")

    # If -1 (clarification needed), look for the second number (task type)
    if classification == -1 and len(numbers) > 1:
        task_id = int(numbers[1])
        logger.debug(f"Clarification needed for task_id: {task_id}")
        return (-1, task_id)

    # Validate the code
    if classification not in [0, 1, 2, 3, 4, -1]:
        logger.warning(f"Invalid classification code {classification}, defaulting to 0")
        return (0, None)

    return (classification, None)


# ==================== Phase 2: Data Extraction Parsing ====================

def parse_arithmetic_data(response_text: str) -> str:
    """Extract a math expression from Phase 2 response.

    Args:
        response_text: Model response containing the expression.

    Returns:
        The mathematical expression string.

    Example:
        ``"2 + 2 * 2"`` → ``"2 + 2 * 2"``
    """
    logger.debug(f"Parsing arithmetic expression: {response_text}")

    expression = response_text.strip()

    # If there are multiple lines, take the first non-empty one
    lines = [line.strip() for line in expression.split('\n') if line.strip()]
    if lines:
        expression = lines[0]

    logger.debug(f"Extracted expression: {expression}")
    return expression


def parse_matrix_data(response_text: str) -> List[List[int]]:
    """Parse a matrix from plain-text lines (for TSP / Max Clique).

    Args:
        response_text: Model response with the matrix rows.

    Returns:
        Matrix as a list of lists of ints.

    Raises:
        ValueError: If the matrix is not square.

    Example:
        ``"0 10 15\\n10 0 20\\n15 20 0"``
        → ``[[0, 10, 15], [10, 0, 20], [15, 20, 0]]``
    """
    logger.debug(f"Parsing matrix data: {response_text}")

    lines = response_text.strip().split('\n')
    matrix: List[List[int]] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        numbers = re.findall(r'-?\d+', line)
        if numbers:
            row = [int(x) for x in numbers]
            matrix.append(row)

    logger.debug(f"Parsed matrix: {matrix}")

    # Validate: matrix must be square
    if matrix:
        size = len(matrix)
        for row in matrix:
            if len(row) != size:
                logger.warning(f"Matrix is not square: {matrix}")
                raise ValueError("Matrix must be square")

    return matrix


def parse_knapsack_data(response_text: str) -> Dict:
    """Parse knapsack data from Phase 2 response.

    Expected format::

        <capacity>
        <weight1> <value1>
        <weight2> <value2>
        ...

    Args:
        response_text: Model response with knapsack data.

    Returns:
        Dict with ``capacity`` (int) and ``items`` (list of dicts).

    Raises:
        ValueError: If the data cannot be parsed.

    Example:
        ``"10\\n2 5\\n3 10"``
        → ``{"capacity": 10, "items": [{"weight": 2, "value": 5}, {"weight": 3, "value": 10}]}``
    """
    logger.debug(f"Parsing knapsack data: {response_text}")

    lines = [line.strip() for line in response_text.strip().split('\n') if line.strip()]

    if not lines:
        raise ValueError("Empty knapsack data")

    # First line — capacity
    capacity_numbers = re.findall(r'\d+', lines[0])
    if not capacity_numbers:
        raise ValueError("No capacity found in first line")

    capacity = int(capacity_numbers[0])

    # Remaining lines — items (weight value)
    items: List[Dict] = []
    for line in lines[1:]:
        numbers = re.findall(r'\d+', line)
        if len(numbers) >= 2:
            weight = int(numbers[0])
            value = int(numbers[1])
            items.append({"weight": weight, "value": value})

    result = {
        "capacity": capacity,
        "items": items
    }

    logger.debug(f"Parsed knapsack data: {result}")
    return result


# ==================== Legacy helpers (still used by solve_task_from_str) ====================

def clean_json_response(text: str) -> str:
    """Remove markdown formatting from JSON response.

    Handles cases like:
    - ```json\\n{...}\\n```
    - ```\\n{...}\\n```
    - Escaped quotes like \\"
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
            inner = json.loads(cleaned)
            if isinstance(inner, str):
                cleaned = inner
                logger.debug("Unwrapped quoted JSON string")
        except json.JSONDecodeError:
            pass

    return cleaned


def is_json_response(text: str) -> bool:
    logger.debug(f"Checking if response is JSON: {text[:100]}{'...' if len(text) > 100 else ''}")

    try:
        json.loads(text)
        logger.debug("Response is valid JSON (raw)")
        return True
    except (json.JSONDecodeError, TypeError):
        pass

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


def _solve_tsp(task_data: dict) -> Result[TaskResult, TaskResult]:
    """Solve TSP using npqtools QUBOSalesman.

    Args:
        task_data: Dict with ``data.distances`` — a square distance matrix.

    Returns:
        Success with route and total distance, or Failure.
    """
    distances = task_data.get("data", {}).get("distances")
    if not distances:
        return Failure(TaskResult("Ошибка: В данных нет поля 'distances'", 1))

    matrix = np.array(distances, dtype=np.int64)
    logger.info(f"TSP distance matrix shape: {matrix.shape}")

    evaluator = SalesmanEvaluator()
    result = evaluator.evaluate(matrix)

    if result is None or result.get("characteristics") is None:
        return Failure(TaskResult("Ошибка: Не удалось найти оптимальный маршрут", 1))

    route = result["characteristics"].tolist()
    total_distance = int(result["answer"])

    route_str = " → ".join(str(city + 1) for city in route)
    route_str += f" → {route[0] + 1}"

    answer = (
        f"Оптимальный маршрут: {route_str}\n"
        f"Общая длина маршрута: {total_distance}"
    )
    logger.info(f"TSP solved: {answer}")
    return Success(TaskResult(answer, 1))


def _solve_clique(task_data: dict) -> Result[TaskResult, TaskResult]:
    """Solve Max Clique using npqtools QUBOClique.

    Args:
        task_data: Dict with ``data.adjacency_matrix`` — a square 0/1 matrix.

    Returns:
        Success with clique vertices and size, or Failure.
    """
    adjacency = task_data.get("data", {}).get("adjacency_matrix")
    if not adjacency:
        return Failure(TaskResult("Ошибка: В данных нет поля 'adjacency_matrix'", 2))

    matrix = np.array(adjacency, dtype=np.int64)
    logger.info(f"Max Clique adjacency matrix shape: {matrix.shape}")

    evaluator = CliqueEvaluator()
    result = evaluator.evaluate(matrix)

    vertices = result["characteristics"].tolist()
    clique_size = int(result["answer"])

    vertices_str = ", ".join(str(v + 1) for v in vertices)

    answer = (
        f"Максимальная клика: вершины {{{vertices_str}}}\n"
        f"Размер клики: {clique_size}"
    )
    logger.info(f"Max Clique solved: {answer}")
    return Success(TaskResult(answer, 2))


def _solve_knapsack(task_data: dict) -> Result[TaskResult, TaskResult]:
    """Solve Knapsack using npqtools QUBOKnapsack.

    Args:
        task_data: Dict with ``data.capacity`` (int) and ``data.items``
            (list of dicts with ``weight`` and ``value``).

    Returns:
        Success with selected items and total value, or Failure.
    """
    data = task_data.get("data", {})
    capacity = data.get("capacity")
    items = data.get("items")

    if capacity is None or not items:
        return Failure(TaskResult("Ошибка: В данных нет 'capacity' или 'items'", 3))

    # QUBOKnapsack expects nx2 array: [cost, weight] per row (int64)
    objects_list = np.array(
        [[item["value"], item["weight"]] for item in items],
        dtype=np.int64,
    )
    logger.info(f"Knapsack: capacity={capacity}, items={len(items)}, objects_list shape={objects_list.shape}")

    evaluator = KnapsackEvaluator()
    result = evaluator.evaluate(objects_list, capacity=capacity)

    selected_indices = result["characteristics"].tolist()
    total_value = int(result["answer"])
    total_weight = int(result["total_weight"])

    selected_items_str = ", ".join(str(i + 1) for i in selected_indices)

    answer = (
        f"Выбранные предметы (номера): {{{selected_items_str}}}\n"
        f"Суммарная ценность: {total_value}\n"
        f"Суммарный вес: {total_weight} (вместимость: {capacity})"
    )
    logger.info(f"Knapsack solved: {answer}")
    return Success(TaskResult(answer, 3))


def solve_task_from_str(json_response_str: str) -> Result[TaskResult, TaskResult]:
    """Parse a JSON string and solve the corresponding task.

    Supports:
    * task_id=1 — TSP (via npqtools QUBOSalesman)
    * task_id=2 — Max Clique (via npqtools QUBOClique)
    * task_id=3 — Knapsack (via npqtools QUBOKnapsack)
    * task_id=4 — Arithmetic (via ``eval``)
    """
    logger.info(f"Solving task from JSON string: {json_response_str}")

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

        elif task_id == 1:
            return _solve_tsp(task_data)

        elif task_id == 2:
            return _solve_clique(task_data)

        elif task_id == 3:
            return _solve_knapsack(task_data)

        else:
            logger.warning(f"Unknown task type: {task_id}")
            return Failure(TaskResult("Неизвестный тип задачи", task_id))

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return Failure(TaskResult("Ошибка: На вход подана некорректная JSON-строка", None))
    except Exception as e:
        logger.error(f"System error while solving task: {e}")
        return Failure(TaskResult(f"Системная ошибка: {e}", None))
