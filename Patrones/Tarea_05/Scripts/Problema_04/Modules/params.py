from numpy import array


def get_params() -> dict:
    params = {
        "path data": "Data",
        "path graphics": "../../Graphics/Problema_04",
        "path results": "../../Results/Problema_04",
        "file data": "data.csv",
        "type cash": {"False": "#660708",
                      "True": "#55a630"
                      },
        "data headers": {"variance": "#f72585",
                         "skewness": "#7209b7",
                         "curtosis": "#3a0ca3",
                         "entropy": "#4361ee",
                         },
        "pair plots": [["variance", "skewness"],
                       ["variance", "curtosis"],
                       ["skewness", "curtosis"]],
        "SVM kernels": ["linear",
                        "poly",
                        "rbf",
                        "sigmoid"
                        ],
    }
    return params


def get_colors_array(params: dict, labels: array) -> array:
    colors = [params["type cash"]["True"]
              if value
              else params["type cash"]["False"]
              for value in labels]
    return colors
