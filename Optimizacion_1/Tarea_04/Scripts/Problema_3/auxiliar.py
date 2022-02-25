from fileinput import filename
from pandas import DataFrame, read_csv


def join_path(path: str, filename: str) -> str:
    return "{}{}".format(path, filename)


def obtain_filename(parameters: dict) -> str:
    filename = "lambda_{}.csv".format(parameters["lambda"])
    if parameters["test data"]:
        filename = "lambda_{}_test.csv".format(parameters["lambda"])
    return filename


def read_data(filename: str) -> DataFrame:
    data = read_csv(filename, index_col=0)
    return data


def write_status(name: str) -> None:
    text = "Resolviendo {}".format(name)
    print("\n")
    print("-"*len(text))
    print(text)


def obtain_graphics_filename(parameters: dict) -> str:
    filename = "lambda_{}.png".format(parameters["lambda"])
    if parameters["test data"]:
        filename = "lambda_{}_test.png".format(parameters["lambda"])
    return filename


def obtain_colors_per_method() -> dict:
    colors = {"x": "#007200",
              "y": "#9d0208",
              }
    return colors


def obtain_filename_for_random_results(dataset: str) -> str:
    filename = filename.split()
    filename = "_".join(filename)
    filename = "{}.csv".format(filename)
    return filename
