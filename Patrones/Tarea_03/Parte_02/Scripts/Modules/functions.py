def obtain_parameters() -> dict:
    parameters = {
        "path data": "Data/",
        "path results": "Results/",
        "path graphics": "Graphics/",
        "path graphics animales": "Graphics/Animals/",
        "matrix file": "predicate-matrix-continuous.txt",
        "classes file": "classes.txt",
        "linkage types": ["single", "complete", "ward"]
    }
    return parameters


def join_path(path: str, filename: str) -> str:
    return "{}{}".format(path, filename)
