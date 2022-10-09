import time


class Stopwatch:
    def __init__(self):
        self._start_time = 0
        self._end_time = 0
        self._message = ""
        self._counter = 0
        self._get_new_sw_counter()

    def tick(self, message: str = "unspecified"):
        self._start_time = time.time()
        self._message = self._convert_to_single_line_str(message)
        # self._counter += 1

        msg = f">>> {self._counter} starting operation: {self._message}"
        print(msg)

    def tock(self):
        self._end_time = time.time()
        message = f"<<< {self._counter} time taken: {self._time_diff()} seconds"
        print(message)

    def _time_diff(self) -> float:
        time_diff = round(self._end_time - self._start_time, 2)

        return time_diff

    @staticmethod
    def _convert_to_single_line_str(message: str) -> str:
        return "\\n".join(message.splitlines())

    @staticmethod
    def _check_sw_counter():
        if "sw_counter" not in globals():
            global sw_counter
            sw_counter = 0

    def _get_new_sw_counter(self):
        self._check_sw_counter()

        global sw_counter
        sw_counter += 1

        self._counter = sw_counter
