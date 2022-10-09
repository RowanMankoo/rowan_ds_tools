# lots of diff write_log functions, but maybe only one logging decorator?
import logging
import json
import functools
import inspect

from ._param_validation import validate_params, StrOptions

config = {"level": "info", "output_file_location": "system.log"}

level_lookup = {"debug": 10, "info": 20, "warning": 30, "error": 40, "critical": 50}

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(level_lookup[config["level"]])
handler = logging.FileHandler(config["output_file_location"], mode="w")
formatter = logging.Formatter("%(asctime)s - %(levelname)s -%(message)s")
handler.setFormatter(formatter)
LOGGER.addHandler(handler)


@validate_params(
    {"level": [StrOptions({"debug", "info", "warning", "error", "critical"})]}
)
def write_basic_log(level="info", error_message=None, **variables_to_log):
    """writes basic log when called

    Args:
        level (str): type of log
        error_message (str, optional): error messsage to record if one has been thrown up
    """

    # Create JSON message to log
    json_message = {"error_message": error_message}
    json_message = json_message | variables_to_log
    json_message = json.dumps(json_message)

    # Log the message
    LOGGER.log(level=level_lookup[level], msg=json_message)


@validate_params({"silent": ["boolean"], "log_varibles": ["boolean"]})
def function_logging_decorator(silent=True, log_varibles=True):
    """Multi-level decorator to log if function runs or not, also sets option to fail in silence
    wraps function around a try/except clause and has option to record all varibles passed to the function
    Args:
        silent (bool): If True then function will not throw error if it fails
        log_varibles (bool): If True then the input varibles to the function will be logged
    """

    # Inner decorator needed so that our decorator can have arguments
    def inner_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            if log_varibles:
                all_func_args = inspect.signature(func).bind(*args, **kwargs).arguments
                all_func_args = dict((k, str(v)) for k, v in all_func_args.items())
            else:
                all_func_args = {}

            # Try run fuction, and log as 'info' if it succeeds at running, along with pre-defined varibles to log **variables_to_log
            try:
                out = func(*args, **kwargs)
                write_basic_log(level="info", **all_func_args)
                return out

            # If function fails log as error and record error message, along with pre-defined varibles to log **variables_to_log
            except Exception as e:
                write_basic_log(level="error", error_message=repr(e), **all_func_args)
                if silent:
                    return
                else:
                    raise

        return wrapper

    return inner_decorator


# def environment_logging_decorator(self, silent=True, **variables_to_log):
#     """Multi-level decorator to log if function runs or not, also sets option to fail in silence
#     Use this decorator if you wish to record specific varibles in environment, and not just variables passed to function

#     Args:
#         silent (bool): If True then function will not throw error if it fails
#         **variables_to_log: set these as varibles in current workspace you wish to record in the logs
#     """

#     # Inner decorator needed so that our decorator can have arguments
#     def inner_decorator(func):

#         @functools.wraps(func)
#         def wrapper(*args,**kwargs):

#             # Try run fuction, and log as 'info' if it succeeds at running, along with pre-defined varibles to log **variables_to_log
#             try:
#                 out = func(*args,**kwargs)
#                 self.write_basic_log(level='info', **variables_to_log)
#                 return out

#             # If function fails log as error and record error message, along with pre-defined varibles to log **variables_to_log
#             except Exception as e:
#                 self.write_basic_log(level='error', error_message=repr(e), **variables_to_log)
#                 if silent:
#                     return
#                 else:
#                     raise

#         return wrapper

#     return inner_decorator
