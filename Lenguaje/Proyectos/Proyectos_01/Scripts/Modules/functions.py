from isort import file


def join_path(path: str, filename: str) -> str:
    """
    Une la direccion de un archivo con su nombre
    """
    return "{}{}".format(path, filename)


def obtain_name_place_from_filename(filename: str) -> str:
    name = filename.replace(".csv", "")
    name = name.split("_")
    name = " ".join(name)
    return name
