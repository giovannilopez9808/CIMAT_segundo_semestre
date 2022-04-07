from Modules.params import get_datasets, obtain_filename_iteration
from Modules.problem import problem_model
from rich.progress import track
from time import sleep


def scrape_data():
    sleep(0.1)


def header(dataset) -> None:
    text = "function {} paso {}".format(dataset["function"],
                                        dataset["step model"])
    return text


# Carga de los datasets
datasets = get_datasets()
for function in datasets["functions"]:
    for step_model in datasets["step models"]:
        # Parametros para inicializar la optimizacion dada una funcion y un tamaño de paso
        dataset = {
            "function": function,
            "step model": step_model,
        }
        text = header(dataset)
        # Realiza un numero de iteraciones dado un paso
        for iteration in track(range(datasets["iteration"]),
                               description='[green]{}'.format(text)):
            # Inicialzacion del problema
            problem = problem_model(dataset)
            filename = obtain_filename_iteration(problem.params,
                                                 dataset,
                                                 iteration+1)
            results = problem.run()
            results.to_csv(filename)
            scrape_data()
