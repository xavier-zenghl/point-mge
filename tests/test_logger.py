import os
import tempfile
from utils.logger import get_logger


def test_logger_creates_log_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = os.path.join(tmpdir, "test.log")
        logger = get_logger("test", log_file=log_path)
        logger.info("hello")
        assert os.path.exists(log_path)
        with open(log_path) as f:
            content = f.read()
        assert "hello" in content


def test_logger_without_file():
    logger = get_logger("console_only")
    logger.info("console message")
