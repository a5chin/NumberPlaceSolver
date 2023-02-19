import logging
from typing import Optional


class Logger:
    def __init__(self, filename: Optional[str] = None) -> None:
        self.filename = filename
        self.fmt = logging.Formatter(
            fmt="[%(asctime)s] :%(name)s: [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        handler.setFormatter(self.fmt)
        self.logger.addHandler(handler)

        if self.filename is not None:
            handler = logging.FileHandler(filename=self.filename)
            handler.setFormatter(self.fmt)
            self.logger.addHandler(handler)

    def log(self, msg: str):
        self.logger.info(msg=msg)
