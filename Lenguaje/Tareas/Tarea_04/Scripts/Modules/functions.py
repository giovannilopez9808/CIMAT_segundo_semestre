from nltk.tokenize import TweetTokenizer
import re
# Tokenizo oraciones


def tokenize(my_txt):
    # Expresión para quitar caracteres
    reg_exp = r"<?/?[A-Z|a-z]+>?"
    tokenizer = TweetTokenizer().tokenize
    # coniverto texto a minúsculas y limpio
    tmp = re.findall(reg_exp, my_txt.lower())
    tmp = ' '.join(tmp)
    # tokenizo
    tmp = tokenizer(tmp)
    return tmp


# def tokenize(text: str):

#     # print(text)
#     tokenizer = TweetTokenizer().tokenize
#     tokens = tokenizer(text)
#     # text = text.lower()
#     # stopwords = list(string.punctuation)
#     # stopwords += [":", "(", ")"]
#     # tokens = tokenizer(text)
#     # tokens = [token for token in tokens
#     #           if not token in stopwords]
#     return tokens


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
