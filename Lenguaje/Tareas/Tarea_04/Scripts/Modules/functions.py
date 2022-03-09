from nltk.tokenize import TweetTokenizer
from unidecode import unidecode
from os import listdir
import re


def tokenize(text: str) -> list:
    """
    Realiza la tokenizacion de un texto dado
    ------------
    Inputs:
    text -> string de texto

    ------------
    Outputs:
    aux -> lista de tokens
    """
    # Expresión para obtener palabras
    reg_exp = r"<?/?[A-Z|a-z]+>?"
    tokenizer = TweetTokenizer().tokenize
    text = unidecode(text)
    # coniverto texto a minúsculas y limpio
    aux = re.findall(reg_exp, text.lower())
    aux = ' '.join(aux)
    # tokenizo
    aux = tokenizer(aux)
    return aux


def ls(path: str) -> list:
    return sorted(listdir(path))


def join_path(path: str, filename: str) -> str:
    """
    Une la direccion de un archivo con su nombre
    """
    return "{}{}".format(path, filename)


def mask_unknow(tweet: str, vocabulary: list) -> str:
    """
    Enmascaramiento de una oración dado un vocabulario
    -----------
    Inputs:
    + tweet -> string con el tweet a enmascarar
    + vocabulary -> vocabulario de los datos de entrenamiento

    ------------
    Outputs:
    tweet_mask -> string con el tweet enmascarado
    """
    # Tokens del tweet dado
    tokens = tokenize(tweet)
    # Enmascaramiento de los tokens
    tweet_mask = [word if word in vocabulary else "<unk>"
                  for word in tokens]
    # Union de los tokens
    tweet_mask = " ".join(tweet_mask)
    return tweet_mask
