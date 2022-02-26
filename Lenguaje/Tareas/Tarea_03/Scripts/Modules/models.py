from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
from sklearn import metrics, svm
from numpy import array, shape


class SVM_model:
    def __init__(self) -> None:
        pass

    def create_model(self, bow_tr: array, labels_tr: array) -> GridSearchCV:
        """
        Creacion del modelo para realizar el aprendizaje
        """
        parameters_model = {"C": [0.05, 0.12, 0.25, 0.5, 1, 2, 4]}
        svr = svm.LinearSVC(class_weight="balanced", max_iter=1200000)
        grid = GridSearchCV(estimator=svr,
                            param_grid=parameters_model,
                            n_jobs=8,
                            scoring="f1_macro",
                            cv=5)
        grid.fit(bow_tr, labels_tr)
        return grid

    def evaluate_model(self, bow_val: array, labels_val: array, grid: GridSearchCV,
                       name: str) -> list:
        """
        Resultados del modelo con el dataset de validacion
        """
        y_pred = grid.predict(bow_val)
        precision, recall, fscore, _ = precision_recall_fscore_support(
            labels_val,
            y_pred,
            average="macro",
            pos_label=1,
        )
        # print(confusion_matrix(
        #     labels_val,
        #     y_pred,
        # ))
        # print(metrics.classification_report(
        #     labels_val,
        #     y_pred,
        # ))
        return [name, precision, recall, fscore]
