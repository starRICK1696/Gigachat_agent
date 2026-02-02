import json
from dataclasses import dataclass
from typing import Optional
from returns.result import Result, Success, Failure


@dataclass
class TaskResult:
    result: str
    task_id: Optional[int]


def is_json_response(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def get_task_summary(task_id: int) -> str:
    task_names: dict[int, str] = {
        1: "TSP",
        2: "Max Clique",
        3: "Knapsack",
        4: "Arithmetic"
    }
    task_name = task_names.get(task_id, "Unknown")
    return f"[Task: {task_name}, id={task_id}]"


def solve_task_from_str(json_response_str: str) -> Result[TaskResult, TaskResult]:
    """
    Принимает JSON-строку, парсит её и решает задачу.
    """
    try:
        task_data = json.loads(json_response_str)
        
        task_id = task_data.get("task_id")
        if task_id == 4:
            expression = task_data.get("data", {}).get("expression")
            if not expression:
                return Failure(TaskResult("Ошибка: В JSON нет поля 'expression'", task_id))
            
            try:
                result = eval(expression)
                return Success(TaskResult(f'{expression} = {result}', task_id))
            except Exception as e:
                return Failure(TaskResult(f"Ошибка вычисления eval: {e}", task_id))

        elif task_id in [1, 2, 3]:
            return Failure(TaskResult(f"Задача типа {task_id} пока не поддерживается", task_id))
        else:
            return Failure(TaskResult("Неизвестный тип задачи", task_id))

    except json.JSONDecodeError:
        return Failure(TaskResult("Ошибка: На вход подана некорректная JSON-строка", None))
    except Exception as e:
        return Failure(TaskResult(f"Системная ошибка: {e}", None))