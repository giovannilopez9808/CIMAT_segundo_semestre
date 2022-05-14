from numpy.random import random
from numpy import array


def get_params(dataset: dict) -> dict:
    params = {
        "path graphics": "Graphics",
        "path results": "Results",
        "path data": "Data",
        "image name": "lenanoise.png",
        "c1": 0.1,
        "c2": 0.9,
        "methods": [
            "FR",
            "PR",
            "HS",
            "FR PR"
        ],
        "lambda values": {
            0.1: {"name": "lambda_1"},
            1: {"name": "lambda_2"},
            10: {"name": "lambda_3"}
        }
    }
    params.update(dataset)
    return params


def get_params_from_image(params: dict, image: array) -> dict:
    params["x"] = random(image.shape)
    params["g"] = image
    params["n"] = int(image.shape[0]**(1/2))
    return params
