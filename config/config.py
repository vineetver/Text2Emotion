import logging
import logging.config
import sys
from pathlib import Path
from rich.logging import RichHandler

BASE_DIR = Path(__file__).parent.parent
LOGS_DIR = Path(BASE_DIR, 'logs')
LOGS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(BASE_DIR, 'data')
MODEL_DIR = Path(BASE_DIR, 'models')
CONFIG_DIR = Path(BASE_DIR, 'config')

TRAIN_URL = 'https://github.com/google-research/google-research/raw/master/goemotions/data/train.tsv'
VALID_URL = 'https://github.com/google-research/google-research/raw/master/goemotions/data/dev.tsv'
TEST_URL = 'https://github.com/google-research/google-research/raw/master/goemotions/data/test.tsv'


logging_config = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'minimal': {'format': '%(message)s'},
        'detailed': {
            'format': '%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'stream': sys.stdout,
            'formatter': 'minimal',
            'level': logging.DEBUG,
        },
        'info': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': Path(LOGS_DIR, 'info.log'),
            'maxBytes': 10485760,
            'backupCount': 10,
            'formatter': 'detailed',
            'level': logging.INFO,
        },
        'error': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': Path(LOGS_DIR, 'error.log'),
            'maxBytes': 10485760,
            'backupCount': 10,
            'formatter': 'detailed',
            'level': logging.ERROR,
        },
    },
    'root': {
        'handlers': ['console', 'info', 'error'],
        'level': logging.INFO,
        'propagate': True,
    },
}

logging.config.dictConfig(logging_config)
logger = logging.getLogger()
logger.handlers[0] = RichHandler(markup=True)
