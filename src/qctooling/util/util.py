
from functools import wraps
from typing import Callable, TypeVar, Union, ParamSpec, List
from time import perf_counter

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

def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            print(f"{func.__name__}: {perf_counter() - start:.3f}s")
    return wrapper