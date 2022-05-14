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
            0.1: {"name": "lambda_1"},
            0.5: {"name": "lambda_2"},
            1: {"name": "lambda_3"}
        }
    }
    params.update(dataset)
    return params


def get_graphics_params() -> dict:
    params = {
        0.1: {"y lim": {"Function": [10000, 34000],
                        "Gradient": [0, 350], },
              "x lim": [0, 30],
              "x delta": 3,
              "y delta": {"Function": 4000,
                          "Gradient": 50}
              },
        0.5:  {"y lim": {"Function": [10000, 400000],
                         "Gradient": [0, 2000], },
               "x lim": [0, 50],
               "x delta": 5,
               "y delta": {"Function": 100000,
                           "Gradient": 250}
               },
        1:  {"y lim": {"Function": [100000, 400000],
                       "Gradient": [0, 3000], },
             "x lim": [0, 50],
             "x delta": 5,
             "y delta": {"Function": 50000,
                         "Gradient": 600}
             },
    }
    return params


def get_params_from_image(params: dict, image: array) -> dict:
    params["x"] = random(image.shape)*2
    params["g"] = image
    params["n"] = int(image.shape[0]**(1/2))
    return params
