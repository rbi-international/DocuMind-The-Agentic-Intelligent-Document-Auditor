from documind import logger
import torch
import sys

try:
    logger.info(">>> TESTING ENVIRONMENT <<<")
    logger.info(f"Python Version: {sys.version}")
    logger.info(f"PyTorch Version: {torch.__version__}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
    logger.info("Local 'documind' package imported successfully.")
except Exception as e:
    logger.error(f"Environment Check Failed: {e}")