import structlog
import logging

def get_logger(level: str, filename: str) -> structlog.BoundLogger:
    structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter
    ],
    logger_factory=structlog.stdlib.LoggerFactory())
    
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer(
            colors=True,
        )
    )
    json_formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer()
    )

    handler_file = logging.FileHandler(filename)
    handler_file.setFormatter(json_formatter)
    handler_cmd = logging.StreamHandler()
    handler_cmd.setFormatter(formatter)
    
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.addHandler(handler_file)
    logger.addHandler(handler_cmd)  
    
    return structlog.get_logger()