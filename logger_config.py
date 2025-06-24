import logging
from logging.handlers import RotatingFileHandler

def setup_logging(name: str = __name__) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler("app.log", maxBytes=1000000, backupCount=5)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    if not logger.handlers:
        logger.addHandler(handler)
    return logger