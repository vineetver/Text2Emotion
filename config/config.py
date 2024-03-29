import logging.config
import sys
from pathlib import Path
from rich.logging import RichHandler
import mlflow

BASE_DIR = Path(__file__).parent.parent
LOGS_DIR = Path(BASE_DIR, 'logs')
DATA_DIR = Path(BASE_DIR, 'data')
CONFIG_DIR = Path(BASE_DIR, 'config')
STORES_DIR = Path(BASE_DIR, 'stores')


MODEL_REGISTRY = Path(STORES_DIR, 'model')
BLOB_STORE = Path(STORES_DIR, 'blob')

MODEL_REGISTRY.mkdir(parents=True, exist_ok=True)
BLOB_STORE.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri("file://" + str(MODEL_REGISTRY.absolute()))

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
