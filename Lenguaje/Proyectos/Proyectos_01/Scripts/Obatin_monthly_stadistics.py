from Modules.functions import join_path, obtain_name_place_from_filename
from Modules.tripadvisor import tripadvisor_model
from Modules.datasets import parameters_model
from os import listdir as ls


dataset = parameters_model()
dataset.parameters["path results"] += "Monthly_mean/"
tripadvisor = tripadvisor_model(dataset)
files = sorted(ls(dataset.parameters["path data"]))
for file in files:
    nameplace = obtain_name_place_from_filename(file)
    filename = join_path(dataset.parameters["path results"],
                         file)
    tripadvisor.read_data(file)
    tripadvisor.obtain_monthly_stadistics_of_scores()
    tripadvisor.monthly_data.to_csv(filename,
                                    float_format="%.4f")
