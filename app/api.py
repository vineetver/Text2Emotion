from datetime import datetime
from functools import wraps

from fastapi import FastAPI, Request
from app.schema import PredictPayload

app = FastAPI(
    title='Text2Emotion - by @vineet_verma',
    description='Detect fine-grained emotion from text',
    version='0.0.1'
)


def construct_response(f):
    """Construct a JSON response for an endpoint"""

    @wraps(f)
    def wrap(request: Request, *args, **kwargs) -> Dict:
        results = f(request, *args, **kwargs)
        response = {
            'message'    : results['message'],
            'method'     : request.method,
            'status-code': results['status-code'],
            'timestamp'  : datetime.now().isoformat(),
            'url'        : request.url.url,
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


# @app.get('/performance', tags=['Performance'])
# @construct_response
# def _performance(request: Request):
#     """Performance check"""
#     performance =

