# mypackage/__init__.py

__version__ = "1.0.0"
__author__ = "Sean Gleason"

from .dinocut import Class1, function1


__all__ = ["Class1", "function1", "Class2", "function2"]

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("mypackage initialized")

# Optional: Code to execute on package import
# from . import subpackage1
# from . import subpackage2
