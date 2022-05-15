from numpy.random import random
from numpy import array


def get_params(dataset: dict = {}) -> dict:
    params = {
        "path graphics": "Graphics",
        "path results": "Results",
        "path data": "Data",
        "image name": "lenanoise.png",
        "methods": {
            "FR": "#003049",
            "PR": "#d62828",
            "HS": "#f77f00",
            "FR PR": "#2a9d8f",
        },
        "lambda values": {
            0.1: {
                "name": "lambda_1"
            },
            0.5: {
                "name": "lambda_2"
            },
            1: {
                "name": "lambda_3"
            },
            5: {
                "name": "lambda_4"
            },
        }
    }
    params.update(dataset)
    return params


def get_graphics_params() -> dict:
    params = {
        0.1: {
            "y lim": {
                "Function": [10000, 20000],
                "Gradient": [0, 300],
            },
            "x lim": [0, 30],
            "x delta": 3,
            "y delta": {
                "Function": 2000,
                "Gradient": 50,
            }
        },
        0.5: {
            "y lim": {
                "Function": [50000, 100000],
                "Gradient": [0, 1200],
            },
            "x lim": [0, 50],
            "x delta": 5,
            "y delta": {
                "Function": 15000,
                "Gradient": 200
            }
        },
        1: {
            "y lim": {
                "Function": [100000, 200000],
                "Gradient": [0, 2400],
            },
            "x lim": [0, 50],
            "x delta": 5,
            "y delta": {
                "Function": 20000,
                "Gradient": 400
            }
        },
        5: {
            "y lim": {
                "Function": [500000, 1000000],
                "Gradient": [0, 10000],
            },
            "x lim": [0, 50],
            "x delta": 5,
            "y delta": {
                "Function": 100000,
                "Gradient": 2000
            }
        },
    }
    return params


def get_params_from_image(params: dict, image: array) -> dict:
    params["x"] = random(image.shape)
    params["g"] = image
    params["n"] = int(image.shape[0]**(1 / 2))
    return params
