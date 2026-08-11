from pydantic import BaseModel, PrivateAttr
from typing import Union
import logging, pathlib

class ParserBase(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    path: pathlib.Path
    baseName: str
    lvprt: int = logging.WARNING

    _logger: logging.Logger = PrivateAttr()

    def model_post_init(self, __context) -> None:
        # unique per instance -> independent level control
        logger_name = f"{self.__class__.__module__}.{self.__class__.__name__}.{self.baseName}"
        self._logger = logging.getLogger(logger_name)
        self._logger.setLevel(self.lvprt)

    @property
    def logger(self) -> logging.Logger:
        return self._logger