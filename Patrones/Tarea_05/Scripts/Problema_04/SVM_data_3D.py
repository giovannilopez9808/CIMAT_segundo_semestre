from sklearn.model_selection import train_test_split
from Modules.dataset import dataset_model, join
from sklearn.metrics import accuracy_score
from Modules.params import get_params
from Modules.svm import SVM_model
from pandas import DataFrame

# Lectura de los parametros
params = get_params()
params["file results"] = "SVM_data_3D.csv"
# Lectura de los datos
dataset = dataset_model(params)
data = dataset.data.copy()
# Etiquedas
labels = data["label"].to_numpy()
# Datos de entrenamiento
train = data.drop(columns="label")
train = train.to_numpy()
# Creacion del modelo de SVM
svm = SVM_model(params)
# Inicialización del guardado de los resultado
results = DataFrame(columns=params["SVM kernels"])
for kernel_name in params["SVM kernels"]:
    svm.create(kernel_name)
    # Split data
    data_train, data_validation, label_train, label_validation = train_test_split(
        train,
        labels,
        train_size=0.9,
    )
    # Ejecucion del modelo SVM
    svm.run(data_train,
            label_train,
            create_grid=False)
    # Predicción de valores de validacion
    predict = svm.preditct(data_validation)
    # Score
    score = accuracy_score(label_validation,
                           predict)
    results.loc[0, kernel_name] = score
filename = join(params["path results"],
                params["file results"])
results.to_csv(filename,
               index=False)
