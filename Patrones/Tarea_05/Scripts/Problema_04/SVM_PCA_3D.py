from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from Modules.dataset import dataset_model, join
from pandas import DataFrame, read_csv
from Modules.params import get_params
from Modules.svm import SVM_model
from numpy import round, sum

# Lectura de los parametros
params = get_params()
params["file PCA"] = "PCA.csv"
params["file results"] = "SVM_PCA_3D.csv"
data_types = [["Component 1", "Component 2"], ["Component 2", "Component 3"],
              ["Component 1", "Component 3"]]
# Lectura de los datos
dataset = dataset_model(params)
filename = join(params["path results"], params["file PCA"])
data = read_csv(filename)
# Etiquedas
labels = data["label"].to_numpy()
# Split data
train = data.drop(columns="label")
train = train.to_numpy()
# Creacion del modelo de SVM
svm = SVM_model(params)
# # Inicialización del guardado de los resultado
results = DataFrame(columns=params["SVM kernels"])
for kernel_name in params["SVM kernels"]:
    svm.create(kernel_name)
    # Datos de entrenamiento
    data_train, data_validation, label_train, label_validation = train_test_split(
        train,
        labels,
        train_size=0.9,
    )
    # Ejecucion del modelo SVM
    svm.run(data_train, label_train, create_grid=False)
    # Predicción de valores de validacion
    predict = svm.preditct(data_validation)
    # Score
    score = accuracy_score(label_validation, predict)
    true_table = confusion_matrix(label_validation,
                                  predict)
    total = sum(true_table, axis=None)
    true_table = true_table/total
    true_table = round(true_table, 3)
    print("-"*20)
    print("Kernel {}".format(kernel_name))
    print(true_table)
    # Guardado de los resultados
    results.loc[0, kernel_name] = score
filename = join(params["path results"], params["file results"])
results.to_csv(filename, index=False)
