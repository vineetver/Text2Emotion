from typing import List

from fastapi import Query
from pydantic import BaseModel, validator


class Text(BaseModel):
    text: str = Query(None, min_length=3, max_length=50)


class PredictPayload(BaseModel):
    texts: List[Text]

    @validator('texts')
    def list_must_not_be_empty(cls, value):
        if not len(value):
            raise ValueError('List must not be empty')
        return value

    class Config:
        schema_extra = {
            'example': {
                'texts': [
                    {'text': 'A Ukrainian woman who escaped Russian assault on Mariupol says troops were targeting apartment buildings as '
                             'if they were playing a computer game'},
                    {'text': 'I often go to parks to walk and distress and enjoy nature'},
                    {'text': 'This is the worst muffin ive ever had'},
                ]
            }
        }
