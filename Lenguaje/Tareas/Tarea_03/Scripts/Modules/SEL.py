from functions import obtain_corpus_emotions, join_path


class SEL:
    def __init__(self) -> None:
        pass

    def load_data(self, parameters: dict, data_tr: list, data_val: list) -> tuple:
        """
        Lectura del dataset de SEL
        """
        sel_path = join_path(
            parameters["path data"],
            parameters["SEL"],
        )
        # Carga de la bolsa de emociones de SEL
        words_emotions, scores = self.build_BoE_from_SEL(sel_path)
        data_tr_sel_emotions = obtain_corpus_emotions(
            data_tr,
            words_emotions,
        )
        data_val_sel_emotions = obtain_corpus_emotions(
            data_val,
            words_emotions,
        )
        return data_tr_sel_emotions, data_val_sel_emotions, scores

    def build_BoE(self, filename: str) -> tuple:
        """
        Creacion de una bolsa de emociones a partir de la base de datos de SEL
        """
        # Apertura del archivo
        with open(filename, "r", encoding='latin-1') as file:
            # Inicializacion de los diccionarios
            words_emotions = dict()
            scores = dict()
            # Salto del header
            for i in range(1):
                next(file)
            # Lectura del archivo
            for line in file:
                # Split de los datos
                data = line.split('\t')
                # Score
                score = float(data[1])
                # Palabra en minusculas
                word = data[0].lower()
                # Si no se ha guardado se guarda
                if not word in words_emotions:
                    words_emotions[word] = data[2].replace("\n", "")
                    scores[word] = score
                # Si ya existe se comprueba que sea el que contiene mayor score
                elif score > scores[word]:
                    words_emotions[word] = data[2].replace("\n", "")
                    scores[word] = score
        return words_emotions, scores
