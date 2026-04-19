import json
import logging
import re
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from returns.result import Result, Success, Failure
import numpy as np

from .compute import SalesmanEvaluator, CliqueEvaluator, KnapsackEvaluator, MaxWeightCliqueEvaluator, ProductionSchedulingEvaluator, MultiKnapEvaluator, Tiling2DimEvaluator

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
    if classification not in [0, 1, 2, 3, 4, 5, 6, 7, 8, -1]:
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


def parse_production_scheduling_data(response_text: str) -> Dict:
    """Parse production scheduling data from Phase 2 response.

    Expected format::

        <row1_col1> <row1_col2> ...
        <row2_col1> <row2_col2> ...
        ...

        <setup_time1> <setup_time2> ...
        <job_value1> <job_value2> ...

    The matrix block and the two vector lines are separated by a blank line.

    Args:
        response_text: Model response with scheduling data.

    Returns:
        Dict with ``matrix`` (list of lists), ``setup_times`` (list),
        ``job_values`` (list).

    Raises:
        ValueError: If the data cannot be parsed.
    """
    logger.debug(f"Parsing production scheduling data: {response_text}")

    lines = response_text.strip().split('\n')

    # Split into blocks separated by blank lines
    blocks: List[List[str]] = []
    current_block: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_block:
                blocks.append(current_block)
                current_block = []
        else:
            current_block.append(stripped)
    if current_block:
        blocks.append(current_block)

    if len(blocks) < 2:
        # Try to parse as: matrix lines, then last two lines are vectors
        all_lines = [l.strip() for l in lines if l.strip()]
        if len(all_lines) < 3:
            raise ValueError("Not enough data for production scheduling (need matrix + 2 vectors)")

        # Last two lines are vectors, everything before is matrix
        vector_lines = all_lines[-2:]
        matrix_lines = all_lines[:-2]

        matrix = []
        for ml in matrix_lines:
            numbers = re.findall(r'-?\d+', ml)
            if numbers:
                matrix.append([int(x) for x in numbers])

        setup_times = [int(x) for x in re.findall(r'-?\d+', vector_lines[0])]
        job_values = [int(x) for x in re.findall(r'-?\d+', vector_lines[1])]
    else:
        # Block 1: matrix, Block 2: vectors (2 lines)
        matrix = []
        for ml in blocks[0]:
            numbers = re.findall(r'-?\d+', ml)
            if numbers:
                matrix.append([int(x) for x in numbers])

        # Vectors can be in block 2 (2 lines) or blocks 2 and 3
        if len(blocks) >= 3:
            setup_times = [int(x) for x in re.findall(r'-?\d+', blocks[1][0])]
            job_values = [int(x) for x in re.findall(r'-?\d+', blocks[2][0])]
        elif len(blocks[1]) >= 2:
            setup_times = [int(x) for x in re.findall(r'-?\d+', blocks[1][0])]
            job_values = [int(x) for x in re.findall(r'-?\d+', blocks[1][1])]
        else:
            raise ValueError("Cannot find setup_times and job_values vectors")

    if not matrix:
        raise ValueError("Empty execution time matrix")
    if not setup_times:
        raise ValueError("Empty setup_times vector")
    if not job_values:
        raise ValueError("Empty job_values vector")

    result = {
        "matrix": matrix,
        "setup_times": setup_times,
        "job_values": job_values
    }

    logger.debug(f"Parsed production scheduling data: {result}")
    return result


def parse_multi_knapsack_data(response_text: str) -> Dict:
    """Parse multi-knapsack data from Phase 2 response.

    Expected format::

        <cap1> <cap2> ...
        <value1> <weight1_1> <weight1_2> ...
        <value2> <weight2_1> <weight2_2> ...
        ...

    First line: capabilities (one per constraint type).
    Remaining lines: one per item — first number is value, rest are weights
    for each constraint type.

    Args:
        response_text: Model response with multi-knapsack data.

    Returns:
        Dict with ``capabilities`` (list of ints) and ``items`` (list of lists).

    Raises:
        ValueError: If the data cannot be parsed.
    """
    logger.debug(f"Parsing multi-knapsack data: {response_text}")

    lines = [line.strip() for line in response_text.strip().split('\n') if line.strip()]

    if len(lines) < 2:
        raise ValueError("Not enough data for multi-knapsack (need capabilities + at least 1 item)")

    # First line — capabilities
    cap_numbers = re.findall(r'-?\d+', lines[0])
    if not cap_numbers:
        raise ValueError("No capabilities found in first line")
    capabilities = [int(x) for x in cap_numbers]

    # Remaining lines — items: [value, weight1, weight2, ...]
    items: List[List[int]] = []
    expected_cols = 1 + len(capabilities)  # value + one weight per constraint
    for line in lines[1:]:
        numbers = re.findall(r'-?\d+', line)
        if numbers:
            row = [int(x) for x in numbers]
            items.append(row)

    if not items:
        raise ValueError("No items found")

    result = {
        "capabilities": capabilities,
        "items": items
    }

    logger.debug(f"Parsed multi-knapsack data: {result}")
    return result


def parse_tiling_2d_data(response_text: str) -> Dict:
    """Parse 2D tiling data from Phase 2 response.

    Expected format::

        <width> <length>
        <rect1_width> <rect1_length>
        <rect2_width> <rect2_length>
        ...

        SEPARATIONS
        <row1> <col1> <row2> <col2>
        ...

        BANNED
        <row> <col>
        ...

    First line: area dimensions (width, length).
    Following lines until blank line or SEPARATIONS/BANNED: rectangles.
    Optional SEPARATIONS block: pairs of cells that cannot belong to the same rectangle.
    Optional BANNED block: cells that cannot be covered.

    Args:
        response_text: Model response with tiling data.

    Returns:
        Dict with ``width``, ``length``, ``rectangles`` (list of [w, l]),
        and optionally ``separations`` and ``banned``.

    Raises:
        ValueError: If the data cannot be parsed.
    """
    logger.debug(f"Parsing 2D tiling data: {response_text}")

    lines = response_text.strip().split('\n')

    # Parse into sections
    area_line = None
    rectangles: List[List[int]] = []
    separations: List[List[List[int]]] = []
    banned: List[List[int]] = []

    current_section = "rectangles"  # start by reading area + rectangles

    for line in lines:
        stripped = line.strip()

        if not stripped:
            continue

        if stripped.upper() == "SEPARATIONS":
            current_section = "separations"
            continue
        elif stripped.upper() == "BANNED":
            current_section = "banned"
            continue

        numbers = re.findall(r'-?\d+', stripped)
        if not numbers:
            continue

        if current_section == "rectangles":
            if area_line is None:
                # First line with numbers is the area dimensions
                if len(numbers) >= 2:
                    area_line = [int(numbers[0]), int(numbers[1])]
                else:
                    raise ValueError("Area dimensions line must have 2 numbers (width length)")
            else:
                # Subsequent lines are rectangles
                if len(numbers) >= 2:
                    rectangles.append([int(numbers[0]), int(numbers[1])])

        elif current_section == "separations":
            if len(numbers) >= 4:
                sep = [[int(numbers[0]), int(numbers[1])], [int(numbers[2]), int(numbers[3])]]
                separations.append(sep)

        elif current_section == "banned":
            if len(numbers) >= 2:
                banned.append([int(numbers[0]), int(numbers[1])])

    if area_line is None:
        raise ValueError("No area dimensions found")
    if not rectangles:
        raise ValueError("No rectangles found")

    result: Dict = {
        "width": area_line[0],
        "length": area_line[1],
        "rectangles": rectangles
    }

    if separations:
        result["separations"] = separations
    if banned:
        result["banned"] = banned

    logger.debug(f"Parsed 2D tiling data: {result}")
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
        4: "Arithmetic",
        5: "Max Weight Clique",
        6: "Production Scheduling",
        7: "Multi-Knapsack",
        8: "2D Tiling"
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


def _solve_max_weight_clique(task_data: dict) -> Result[TaskResult, TaskResult]:
    """Solve Max Weight Clique using npqtools QUBOMaxWeightClique.

    Args:
        task_data: Dict with ``data.adjacency_matrix`` — a square weighted matrix
            where values > 0 represent edge weights and 0 means no edge.

    Returns:
        Success with clique vertices and total edge weight, or Failure.
    """
    adjacency = task_data.get("data", {}).get("adjacency_matrix")
    if not adjacency:
        return Failure(TaskResult("Ошибка: В данных нет поля 'adjacency_matrix'", 5))

    matrix = np.array(adjacency, dtype=np.int64)
    logger.info(f"Max Weight Clique adjacency matrix shape: {matrix.shape}")

    evaluator = MaxWeightCliqueEvaluator()
    result = evaluator.evaluate(matrix)

    vertices = result["characteristics"].tolist()
    total_weight = int(result["answer"])

    vertices_str = ", ".join(str(v + 1) for v in vertices)

    answer = (
        f"Клика максимального веса: вершины {{{vertices_str}}}\n"
        f"Суммарный вес рёбер клики: {total_weight}"
    )
    logger.info(f"Max Weight Clique solved: {answer}")
    return Success(TaskResult(answer, 5))


def _solve_production_scheduling(task_data: dict) -> Result[TaskResult, TaskResult]:
    """Solve Production Scheduling using npqtools QUBOProductionScheduling.

    Args:
        task_data: Dict with ``data.matrix`` (2D execution time matrix),
            ``data.setup_times`` (1D setup time vector),
            ``data.job_values`` (1D job value vector).

    Returns:
        Success with scheduling result and total value, or Failure.
    """
    data = task_data.get("data", {})
    matrix = data.get("matrix")
    setup_times = data.get("setup_times")
    job_values = data.get("job_values")

    if matrix is None or setup_times is None or job_values is None:
        return Failure(TaskResult(
            "Ошибка: В данных нет полей 'matrix', 'setup_times' или 'job_values'", 6
        ))

    matrix_np = np.array(matrix, dtype=np.int64)
    setup_times_np = np.array(setup_times, dtype=np.int64)
    job_values_np = np.array(job_values, dtype=np.int64)

    logger.info(
        f"Production Scheduling: matrix shape={matrix_np.shape}, "
        f"setup_times={setup_times_np.tolist()}, job_values={job_values_np.tolist()}"
    )

    evaluator = ProductionSchedulingEvaluator()
    result = evaluator.evaluate(matrix_np, setup_times=setup_times_np, job_values=job_values_np)

    schedule = result["characteristics"].tolist()
    total_value = result["answer"]

    schedule_str = ", ".join(str(s + 1) for s in schedule)

    answer = (
        f"Оптимальное расписание (порядок заказов): [{schedule_str}]\n"
        f"Целевое значение: {total_value}"
    )
    logger.info(f"Production Scheduling solved: {answer}")
    return Success(TaskResult(answer, 6))


def _solve_multi_knapsack(task_data: dict) -> Result[TaskResult, TaskResult]:
    """Solve Multi-Knapsack using npqtools QUBOMultiKnapsack.

    Args:
        task_data: Dict with ``data.items`` (list of lists where each row is
            [value, weight1, weight2, ...]) and ``data.capabilities``
            (list of capacity limits, one per constraint type).

    Returns:
        Success with selected items and total value, or Failure.
    """
    data = task_data.get("data", {})
    items = data.get("items")
    capabilities = data.get("capabilities")

    if items is None or capabilities is None:
        return Failure(TaskResult("Ошибка: В данных нет 'items' или 'capabilities'", 7))

    # QUBOMultiKnapsack expects matrix Nx(1+M): [value, weight1, weight2, ...]
    matrix = np.array(items, dtype=np.int64)
    caps = np.array(capabilities, dtype=np.int64)

    logger.info(
        f"Multi-Knapsack: matrix shape={matrix.shape}, capabilities={caps.tolist()}"
    )

    evaluator = MultiKnapEvaluator()
    result = evaluator.evaluate(matrix, capabilities=caps)

    selected_indices = result["characteristics"].tolist()
    total_value = result["answer"]

    selected_items_str = ", ".join(str(i + 1) for i in selected_indices)

    answer = (
        f"Выбранные предметы (номера): {{{selected_items_str}}}\n"
        f"Суммарная ценность: {total_value}"
    )
    logger.info(f"Multi-Knapsack solved: {answer}")
    return Success(TaskResult(answer, 7))


def _solve_tiling_2d(task_data: dict) -> Result[TaskResult, TaskResult]:
    """Solve 2D Tiling using npqtools QUBOTiling2Dim.

    Args:
        task_data: Dict with ``data.rectangles`` (list of [width, length]),
            ``data.width`` (int), ``data.length`` (int),
            and optionally ``data.separations`` and ``data.banned``.

    Returns:
        Success with tiling result and covered cells count, or Failure.
    """
    data = task_data.get("data", {})
    rectangles = data.get("rectangles")
    width = data.get("width")
    length = data.get("length")
    separations = data.get("separations")
    banned = data.get("banned")

    if rectangles is None or width is None or length is None:
        return Failure(TaskResult(
            "Ошибка: В данных нет полей 'rectangles', 'width' или 'length'", 8
        ))

    matrix = np.array(rectangles, dtype=np.int64)

    logger.info(
        f"2D Tiling: matrix shape={matrix.shape}, width={width}, length={length}, "
        f"separations={separations}, banned={banned}"
    )

    evaluator = Tiling2DimEvaluator()
    result = evaluator.evaluate(
        matrix, width=width, length=length,
        banned=banned, separations=separations
    )

    tiling = result["characteristics"]
    covered_cells = int(result["answer"])

    # Format the tiling grid for display
    tiling_lines = []
    if hasattr(tiling, 'tolist'):
        tiling_list = tiling.tolist()
    else:
        tiling_list = tiling

    for row in tiling_list:
        tiling_lines.append(" ".join(str(cell) for cell in row))
    tiling_str = "\n".join(tiling_lines)

    answer = (
        f"Замощение (номера прямоугольников в каждой клетке):\n{tiling_str}\n"
        f"Покрыто клеток: {covered_cells}"
    )
    logger.info(f"2D Tiling solved: covered_cells={covered_cells}")
    return Success(TaskResult(answer, 8))


def solve_task_from_str(json_response_str: str) -> Result[TaskResult, TaskResult]:
    """Parse a JSON string and solve the corresponding task.

    Supports:
    * task_id=1 — TSP (via npqtools QUBOSalesman)
    * task_id=2 — Max Clique (via npqtools QUBOClique)
    * task_id=3 — Knapsack (via npqtools QUBOKnapsack)
    * task_id=4 — Arithmetic (via ``eval``)
    * task_id=5 — Max Weight Clique (via npqtools QUBOMaxWeightClique)
    * task_id=6 — Production Scheduling (via npqtools QUBOProductionScheduling)
    * task_id=7 — Multi-Knapsack (via npqtools QUBOMultiKnapsack)
    * task_id=8 — 2D Tiling (via npqtools QUBOTiling2Dim)
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

        elif task_id == 5:
            return _solve_max_weight_clique(task_data)

        elif task_id == 6:
            return _solve_production_scheduling(task_data)

        elif task_id == 7:
            return _solve_multi_knapsack(task_data)

        elif task_id == 8:
            return _solve_tiling_2d(task_data)

        else:
            logger.warning(f"Unknown task type: {task_id}")
            return Failure(TaskResult("Неизвестный тип задачи", task_id))

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return Failure(TaskResult("Ошибка: На вход подана некорректная JSON-строка", None))
    except Exception as e:
        logger.error(f"System error while solving task: {e}")
        return Failure(TaskResult(f"Системная ошибка: {e}", None))
