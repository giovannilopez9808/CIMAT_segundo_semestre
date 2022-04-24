from Modules.params import get_colors_array, get_params
from sklearn.model_selection import train_test_split
from Modules.dataset import dataset_model, join
from sklearn.metrics import accuracy_score
from matplotlib.lines import Line2D
from Modules.svm import SVM_model
import matplotlib.pyplot as plt
from pandas import DataFrame

# Lectura de los parametros
params = get_params()
params["file results"] = "SVM_data.csv"
params["file graphics"] = "SVM_data.png"
# Lectura de los datos
dataset = dataset_model(params)
data = dataset.data.copy()
# Creacion del modelo de SVM
svm = SVM_model(params)
fig, axs_list = plt.subplots(4, 3,
                             figsize=(16, 20))
# Inicialización del guardado de los resultado
results = DataFrame(columns=params["SVM kernels"])
for axs, kernel_name in zip(axs_list, params["SVM kernels"]):
    svm.create(kernel_name)
    for i, (ax, data_types) in enumerate(zip(axs, params["pair plots"])):
        data_type1, data_type2 = data_types
        # Datos de entrenamiento
        train = dataset.select_data(data_types)
        # Etiquedas
        labels = dataset.get_labels()
        # Split data
        data_train, data_validation, label_train, label_validation = train_test_split(
            train,
            labels,
            train_size=0.9,
        )
        # Ejecucion del modelo SVM
        svm.run(data_train,
                label_train)
        # Predicción de valores de validacion
        predict = svm.preditct(data_validation)
        # Score
        score = accuracy_score(label_validation,
                               predict)
        colors = get_colors_array(params,
                                  label_train)
        support_vectors = svm.get_suport_vectors()
        # Guardado de los resultados
        results.loc[i, kernel_name] = score
        ax.set_title(kernel_name)
        ax.scatter(svm.grid[0],
                   svm.grid[1],
                   c=svm.colors,
                   alpha=0.2)
        ax.scatter(data_train[:, 0],
                   data_train[:, 1],
                   c=colors,
                   alpha=0.7)
        # ax.scatter(support_vectors[0],
        #            support_vectors[1],
        #            s=200,
        #            linewidth=0.1,
        #            facecolors='none',
        #            edgecolors='black')
        ax.set_xlabel(data_type1)
        ax.set_ylabel(data_type2)
        ax.set_xticks([])
        ax.set_yticks([])
custom_points = [
    Line2D([0], [0],
           marker="o",
           ls="",
           color=color,
           label=label)
    for label, color in params["type cash"].items()]
fig.legend(handles=custom_points,
           loc="upper center",
           ncol=2,
           frameon=False)
plt.tight_layout(pad=2.5)
filename = join(params["path graphics"],
                params["file graphics"])
plt.savefig(filename,
            dpi=400)
filename = join(params["path results"],
                params["file results"])
results.to_csv(filename,
               index=False)
