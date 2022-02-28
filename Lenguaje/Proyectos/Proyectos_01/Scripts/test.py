from Modules.datasets import parameters_model
from Modules.data import tripadvisor_model
from os import listdir as ls

dataset = parameters_model()
tripadvisor = tripadvisor_model(dataset)
files = sorted(ls(dataset.parameters["path data"]))
file = files[0]
tripadvisor.read_data(file)
