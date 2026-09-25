
from functools import wraps
from typing import Callable, TypeVar, Union, ParamSpec, List

P = ParamSpec("P")
T = TypeVar("T")

def catch_errors(func: Callable[P, T]) -> Callable[P, Union[T, Exception]]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Union[T, Exception]:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return e
    return wrapper