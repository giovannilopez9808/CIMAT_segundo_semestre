from Modules.functions import obtain_name_place_from_filename, ls
from Modules.Graphics import plot_ages_histogram
from Modules.datasets import parameters_model
from Modules.tripadvisor import tripadvisor_model


parameters = {"filename": ""}
dataset = parameters_model()
dataset.parameters["path graphics"] += "Histogram/ages_"
tripadvisor = tripadvisor_model(dataset)
files = ls(dataset.parameters["path data"])
for file in files:
    nameplace = obtain_name_place_from_filename(file)
    tripadvisor.read_data(file)
    filename = file.replace(".csv", ".png")
    parameters["filename"] = filename
    plot_ages_histogram(tripadvisor.data["Edad"],
                        dataset,
                        parameters)
