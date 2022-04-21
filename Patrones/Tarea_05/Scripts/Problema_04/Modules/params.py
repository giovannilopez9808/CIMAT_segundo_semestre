def get_params() -> dict:
    params = {
        "path data": "Data",
        "path graphics": "../../Graphics/Problema_04",
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
                       ["skewness", "curtosis"]]
    }
    return params
