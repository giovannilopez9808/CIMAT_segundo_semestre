from functions import *


def write_table(data: pd.DataFrame, sets: list) -> None:
    template = "{} & {} & {} & {} \\\\ \\hline"
    print("-"*40)
    for i, set in enumerate(sets):
        data_set = data[data.index == str(set)]
        data_set = data_set.round(4)
        print(template.format(i+1,
                              data_set["PC1"][str(set)],
                              data_set["PC2"][str(set)],
                              data_set["PC3"][str(set)]))
    print("-"*40)


parameters = {"path data": "../../Results/Problema_3_1/",
              "file loadings": "pca_components.csv"}

data = read_data(parameters["path data"],
                 parameters["file loadings"],
                 use_index=True)
data = data.transpose()
sets_2d = obtain_sets_2d()
sets_3d = obtain_sets_3d(only_numbers=True)
print("Set 2d")
write_table(data, sets_2d)
print("Set 3d")
write_table(data, sets_3d)
