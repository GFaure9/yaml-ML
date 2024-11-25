from loguru import logger
import sys
import time
import threading
from functools import wraps


logger.remove()

FORMAT = " ".join([
    "<green>{time}</green>",
    "<level>| {level} |</level>",
    "<light-black>[{module}]</light-black>",
    "<white>{message}</white>",
])

logger.add(sys.stdout, level="INFO", colorize=True, format=FORMAT)
# logger.add("debug.log", level="DEBUG", colorize=False, format=FORMAT)


def with_spinner(style: str = None):
    """
    A decorator that displays a spinning indicator while the wrapped function is running.

    Parameters
    ----------
    style: str
        Style of loading spinner. See `spinner_styles` below for options.
    """

    spinner_styles = {
        "dots": [".", "..", "..."],
        "spin": r'|/-\\',
        "moon": ['🌑', '🌒', '🌓', '🌔', '🌕', '🌖', '🌗', '🌘'],
        "bouncing": ['(o    )', '( o   )', '(  o  )', '(   o )', '(    o)', '(   o )', '(  o  )', '( o   )'],
        "arrows": ['➡', '↗', '⬆', '↖', '⬅', '↙', '⬇', '↘'],
    }

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):

            # Spinner function
            def spinner():
                while not STOP_SPINNER:
                    for char in spinner_styles[style]:
                        print(f'\r{char} Running {func.__name__}...', end='', flush=True)
                        time.sleep(0.2)

            # Start spinner in a separate thread
            global STOP_SPINNER
            STOP_SPINNER = False
            spinner_thread = threading.Thread(target=spinner)
            spinner_thread.start()

            try:
                # Execute the wrapped function
                result = func(*args, **kwargs)
            finally:
                # Stop the spinner
                STOP_SPINNER = True
                spinner_thread.join()
                print(f'\r✔ {func.__name__} completed!', flush=True)

            return result

        return wrapper

    return decorator
