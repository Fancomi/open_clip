import logging
from datetime import datetime, timezone, timedelta

# Beijing timezone (UTC+8)
BEIJING_TZ = timezone(timedelta(hours=8))


class BeijingFormatter(logging.Formatter):
    """Custom formatter that uses Beijing time (UTC+8) regardless of system timezone."""
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=BEIJING_TZ)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.isoformat()


def setup_logging(log_file, level, include_host=False):
    if include_host:
        import socket
        hostname = socket.gethostname()
        formatter = BeijingFormatter(
            f'%(asctime)s |  {hostname} | %(levelname)s | %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')
    else:
        formatter = BeijingFormatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')

    logging.root.setLevel(level)
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logging.root.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)

