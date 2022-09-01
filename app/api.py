from datetime import datetime
from functools import wraps
from http import HTTPStatus
from pathlib import Path
from fastapi import FastAPI, Request
from app.schema import PredictPayload
from config import config
from src.model import optimization
from config.config import logger
from src.model.classifier import BERT

app = FastAPI(
    title='Text2Emotion - by @vineet_verma',
    description='Detect fine-grained emotion from text',
    version='0.0.1'
)


@app.on_event('startup')
def load_artifacts():
    global artifacts
    run_id = open(Path(config.CONFIG_DIR, 'run_id.txt')).read()
    artifacts = optimization.load_artifacts(run_id=run_id)
    logger.info('✅ Artifacts loaded ✅')
    logger.info(f'✅ Run ID: {run_id} ✅')
    logger.info(f'✅ Ready for inference ✅')


def construct_response(f):
    """Construct a JSON response for an endpoint"""
    @wraps(f)
    def wrap(request: Request, *args, **kwargs) -> dict:
        results = f(request, *args, **kwargs)
        response = {
            'message': results['message'],
            'method': request.method,
            'status-code': results['status-code'],
            'timestamp': datetime.now().isoformat(),
            'url': request.url._url,
        }
        if 'data' in results:
            response['data'] = results['data']
        return response

    return wrap


@app.get('/', tags=['General'])
@construct_response
def _index(request: Request):
    """Status check"""
    response = {
        'message': 'Welcome to Text2Emotion',
        'status-code': 200,
        'data': {}
    }
    return response


@app.get('/performance', tags=['Performance'])
@construct_response
def _performance(request: Request):
    """Performance check"""
    performance = artifacts['metrics']
    data = {'performance': performance.get(filter, performance)}
    response = {
        'message': HTTPStatus.OK.phrase,
        'status-code': HTTPStatus.OK,
        'data': data
    }

    return response


@app.get('/params', tags=['Parameters'])
@construct_response
def _params(request: Request):
    """Parameters check"""
    response = {
        'message': HTTPStatus.OK.phrase,
        'status-code': HTTPStatus.OK,
        'data': {
            'params': vars(artifacts['params'])
        }
    }

    return response


@app.get('/params/{param}', tags=['Parameters'])
@construct_response
def _param(request: Request, param: str):
    """Get a specific parameter's value"""
    response = {
        'message': HTTPStatus.OK.phrase,
        'status-code': HTTPStatus.OK,
        'data': {
            param: vars(artifacts['params']).get(param, "")
        }
    }

    return response


@app.post('/predict', tags=['Predict'])
@construct_response
def _predict(request: Request, payload: PredictPayload):
    """Predict emotion from text"""
    texts = [item.text for item in payload.texts]

    # Load weights
    bert = BERT(params=artifacts['params'])
    model = bert.model
    model.load_weights(Path(artifacts['artifacts_dir'], 'bert_model.hdf5'))

    # Predict
    pred, prob = bert.prediction(prompt=texts, threshold=artifacts['params'].threshold, model=model)
    logger.info(f'🏁 Prediction: {pred} and probability {prob} 🏁')

    pred1 = pred.values.tolist()
    prob1 = prob.values.tolist()
    response = {
        'message': HTTPStatus.OK.phrase,
        'status-code': HTTPStatus.OK,
        'data': {
            'prediction': {
                'pred': pred1,
                'prob': prob1
            }
        }
    }

    return response
