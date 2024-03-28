import sys
import logging

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from logging.handlers import RotatingFileHandler
from config.logging_cfg import LoggingConfig

class Logger:
    def __init__(self, name="", log_level=logging.INFO, log_file=None) -> None:
        self.log = logging.getLogger(name)
        self.get_logger(log_level, log_file)

    def get_logger(self, log_level, log_file):
        self.log.setLevel(log_level)
        self.__init_formatter()
        if log_file is not None:
            self.__add_file_handler(LoggingConfig.LOG_DIR / log_file)
        else:
            self.__add_stream_handler()

    def __init_formatter(self):
        self.formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    def __add_file_handler(self, param):
        file_handler = RotatingFileHandler(param, maxBytes=1024 * 1024 * 10, backupCount=10)
        file_handler.setFormatter(self.formatter)
        self.log.addHandler(file_handler)

    def __add_stream_handler(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(self.formatter)
        self.log.addHandler(stream_handler)

    def log_model(self, predictor_name):
        self.log.info(f"Predictor name: {predictor_name}")

    def log_response(self, pred_prob, pred_id, pred_class):
        self.log.info(
            f"Predicted probability: {pred_prob}\n"
            f"Predicted id: {pred_id}\n"
            f"Predicted class: {pred_class}"
        )