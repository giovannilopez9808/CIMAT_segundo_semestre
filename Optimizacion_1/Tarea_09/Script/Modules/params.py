from numpy.random import random
from numpy import array


def get_params(dataset: dict = {}) -> dict:
    params = {
        "path graphics": "Graphics",
        "path results": "Results",
        "path data": "Data",
        "image name": "lenanoise.png",
        "methods": {
            "FR": "#264653",
            "PR": "#06d6a0",
            "HS": "#dc2f02",
            "FR PR": "#f72585",
        },
        "lambda values": {
            0.1: {"name": "lambda_1"},
            1: {"name": "lambda_2"},
            10: {"name": "lambda_3"}
        }
    }
    params.update(dataset)
    return params


def get_graphics_params() -> dict:
    params = {
        0.1: {"y lim": {"Function": [11000, 22000],
                        "Gradient": [0, 350], },
              "x lim": [0, 50],
              "x delta": 5,
              "y delta": {"Function": 2000,
                          "Gradient": 50}
              },
        1:  {"y lim": {"Function": [100000, 220000],
                       "Gradient": [0, 3000], },
             "x lim": [0, 50],
             "x delta": 5,
             "y delta": {"Function": 20000,
                         "Gradient": 500}
             },
        10:  {"y lim": {"Function": [1000000, 2000000],
                        "Gradient": [0, 20000], },
              "x lim": [0, 50],
              "x delta": 5,
              "y delta": {"Function": 200000,
                          "Gradient": 5000}
              },
    }
    return params


def get_params_from_image(params: dict, image: array) -> dict:
    params["x"] = random(image.shape)
    params["g"] = image
    params["n"] = int(image.shape[0]**(1/2))
    return params
