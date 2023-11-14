import time
import concurrent.futures as pool


def execution_time(func):
    "Декоратор считающий время выполнения функции"
    def wrapper(*args, **kwargs):
        s_t = time.time()
        result = func(*args, **kwargs)
        print("Время выполнения функции %s = " % (func.__name__),
              time.time() - s_t, "сек.")
        return result
    return wrapper


def run_as_multithread(func):
    """Запускает задачу мультипоточно
    Args:
        func: функция
        input_data: вводные дынне функции
    Returns:
        iterable: результат выполнения функции
    """
    def wrapper(*args, **kwargs):
        try:
            with pool.ThreadPoolExecutor() as executer:
                result = list()
                for data in kwargs.get("input_data", []):
                    future = executer.submit(func, *(*args, data), **kwargs)
                    result.append(future)
            return tuple(res.result() for res in pool.as_completed(result))
        except:
            result = list()
            for data in kwargs.get("input_data", []):
                result.append(func(*(*args, data), **kwargs))
            return result
    return wrapper
