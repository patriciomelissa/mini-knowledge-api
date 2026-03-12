import logging


def setup_logging() -> logging.Logger:
    """
    Configure and initialize application logging.

    This function sets up the global logging configuration used by the
    RAG system. It defines the logging level and the message format,
    then returns a dedicated logger instance for the application.

    Log format:
        timestamp | level | logger_name | message

    Returns:
        logging.Logger: Configured logger instance named "rag".
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    return logging.getLogger("rag")
