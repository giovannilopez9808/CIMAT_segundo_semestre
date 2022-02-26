from functions import obtain_corpus_emotions, join_path


class emolex:

    def __init__(self) -> None:
        pass

    def load_emolex_data(self, parameters: dict, data_tr: list, data_val: list) -> tuple:
        """
        Lectura del dataset de EmoLex
        """
        emolex_path = join_path(
            parameters["path data"],
            parameters["EmoLex"],
        )
        # Carga de la bolsa de emociones
        words_emotions = self.build_BoE(emolex_path)
        data_tr_emolex_emotions = obtain_corpus_emotions(
            data_tr,
            words_emotions,
        )
        data_val_emolex_emotions = obtain_corpus_emotions(
            data_val,
            words_emotions,
        )
        return data_tr_emolex_emotions, data_val_emolex_emotions

    def build_BoE(self, filename: str) -> dict:
        """
        Creacion de una bolsa de emociones a partir de la base de datos de EmoLex
        """
        with open(filename, "r", encoding='utf-8') as file:
            # Inicializacion de los diccionarios
            words_dict = dict()
            scores = dict()
            # Salto del header
            for i in range(1):
                next(file)
            for line in file:
                # Lectura de la informacion
                data = line.split('\t')
                if data[1] != 'NO TRANSLATION':
                    # Obtencion del score
                    score = float(data[3])
                    word = data[1].lower()
                    if not word in words_dict:
                        words_dict[word] = data[2]
                        scores[word] = score
                    elif score > scores[word]:
                        words_dict[word] = data[2]
                        scores[word] = score
        return words_dict
