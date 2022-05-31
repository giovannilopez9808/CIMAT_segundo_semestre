from pandas import read_csv, DataFrame, concat
from os import listdir as ls

headers = ["Título de la opinión", "Opinión"]
data = DataFrame()
files = ls()
for file in files:
    if "csv" in file:
        data_file = read_csv(file)
        for header in headers:
            data_file[header] = data_file[header].apply(
                lambda x: x.replace('"', ""))
        data_file["Text"] = data_file[headers].apply(lambda x: " ".join(x),
                                                     axis=1)
        data = concat([data, data_file["Text"]])
data.columns = ["Text"]
data.to_csv("test.csv", index=False)
