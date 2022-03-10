def obtain_parameters() -> dict:
    parameters = {"path data": "../Data/",
                  "path graphics": "../Graphics/",
                  "path graphics animales": "../Graphics/Animals/",
                  "matrix file": "predicate-matrix-continuous.txt",
                  "classes file": "classes.txt"}
    return parameters


def join_path(path: str, filename: str) -> str:
    return "{}{}".format(path, filename)
