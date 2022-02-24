from fileinput import filename
from pandas import DataFrame, read_csv


def join_path(path: str, filename: str) -> str:
    return "{}{}".format(path, filename)


def obtain_filename(parameters: dict) -> str:
    algorithm = parameters["algorithm name"].replace(" ", "_")
    return "{}_{}_{}_{}.csv".format(parameters["problem name"],
                                    parameters["n"],
                                    parameters["initial point"],
                                    algorithm)


def obtain_files_per_dataset(files: list, parameters: dict) -> list:
    dataset = "{}_{}_{}".format(parameters["dataset"],
                                parameters["n"],
                                parameters["initial point"])
    files_function = [file for file in files if dataset in file]
    return files_function


def read_data(filename: str) -> DataFrame:
    data = read_csv(filename, index_col=0)
    return data


def write_status(name: str) -> None:
    text = "Resolviendo {}".format(name)
    print("\n")
    print("-"*len(text))
    print(text)


def obtain_method_from_name(filename: str) -> str:
    method = filename.split("_")[-1]
    method = method.split(".")[0]
    if method == "gradient":
        method += " descent"
    method = method.capitalize()
    return method


def obtain_graphics_filename(parameters: dict) -> str:
    filename = "{}_{}_{}.png".format(parameters["dataset"],
                                     parameters["n"],
                                     parameters["initial point"])
    return filename


def obtain_colors_per_method() -> dict:
    colors = {"Newton": "#007200",
              "Gradient descent": "#9d0208",
              }
    return colors


def obtain_filename_for_random_results(dataset: str) -> str:
    filename = dataset.replace("random", "")
    filename = filename.split()
    filename = "_".join(filename)
    filename = "{}.csv".format(filename)
    return filename
