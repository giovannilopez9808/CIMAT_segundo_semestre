from pandas import DataFrame, read_csv


def join_path(path: str, filename: str) -> str:
    return "{}{}".format(path, filename)


def obtain_filename(parameters: dict) -> str:
    return "{}_{}.csv".format(parameters["problem name"],
                              parameters["lambda"])


def obtain_files_per_function(files: list, parameters: dict) -> list:
    files_function = [file for file in files if parameters["function"] in file]
    return files_function


def read_data(filename: str) -> DataFrame:
    data = read_csv(filename, index_col=0)
    return data
