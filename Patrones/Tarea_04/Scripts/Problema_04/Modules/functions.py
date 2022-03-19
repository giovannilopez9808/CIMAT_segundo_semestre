from os import listdir


def join_path(path: str, name: str) -> str:
    return "{}{}".format(path, name)


def ls(path: str) -> list:
    return sorted(listdir(path))
