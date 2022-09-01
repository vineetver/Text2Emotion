from src.utils import write_dict, get_dict


def test_write_dict():

    dict = {
        "overall": {
            "precision": 0.6107974392471712,
            "recall": 0.7071843931720128,
            "f1": 0.6526798728071054,
            "num_samples": 4884.0
        },
        "classes": {
            "anger": {
                "precision": 0.41578947368421054,
                "recall": 0.5214521452145214,
                "f1": 0.4626647144948755,
                "num_samples": 4884.0
            },
            "disgust": {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "num_samples": 4884.0
            },
            "fear": {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "num_samples": 4884.0
            },
            "joy": {
                "precision": 0.7813233223838574,
                "recall": 0.8516624040920716,
                "f1": 0.814977973568282,
                "num_samples": 4884.0
            },
            "sadness": {
                "precision": 0.5661538461538461,
                "recall": 0.5183098591549296,
                "f1": 0.5411764705882351,
                "num_samples": 4884.0
            },
            "surprise": {
                "precision": 0.55409836065573772232323293232323232323232323219839123912831231231,
                "recall": 0.5348101265822784,
                "f1": 0.5442834138486312,
                "num_samples": 4884.0
            },
            "neutral": {
                "precision": None,
                "recall": 0.7869565217391304,
                "f1": 0.6643943366544309,
                "num_samples": 4884.0
            }
        }
    }

    write_dict(dict, 'test.json')
    assert get_dict('test.json') == dict
