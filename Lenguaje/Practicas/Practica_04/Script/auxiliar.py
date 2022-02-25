from tabulate import tabulate


def join_path(path: str, filename: str) -> str:
    """
    Une la direccion de un archivo con su nombre
    """
    return "{}{}".format(path, filename)


def print_results(results: list) -> None:
    """
    Impresion estandarizada de los resultados
    """
    print(
        tabulate(
            results,
            headers=[
                'Algoritmo',
                'Precision',
                'Recall',
                'F1 Score',
            ],
        ))
