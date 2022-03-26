from Modules.datasets import init_seeds, get_args, get_params
from Modules.models import Mex_data_class, ngram_model, neural_language_model, model_class, generate_text_class
from nltk.tokenize import TweetTokenizer as tokenizer
from Modules.word2vec import word2vec_class
# Semillas de las funciones aleatorias
init_seeds()
# Recoleccion de los parametros y argumentos
params = get_params()
args = get_args()
# Definicion del tokenizer
tokenize = tokenizer().tokenize
print("Lectura de archivos")
# Lectura de los datos
mex_data = Mex_data_class(params, args)
# Lectura de word2vec embeddings
word2vec = word2vec_class(params)
# Inicializacion del modelo de ngramas
ngram = ngram_model(args.N,
                    tokenize=tokenize,
                    embeddings_model=word2vec)
ngram.fit(mex_data.train_text)
# Argumento del tamaño del vocabulario
args.vocabulary_size = ngram.get_vocabulary_size()
# # Estructuración de los datos para la red neuronal
mex_data.obtain_data_and_labels(ngram)
mex_data.obtain_loaders()
# Inicializacion de la red neuronal
neural_model = neural_language_model(args)
# Inicializacion del modelo de prediccion
model = model_class(neural_model,
                    args,
                    mex_data.train_loader,
                    mex_data.validation_loader)
# Entrenamiento de la neurona
# model.run()
# Lectura de los parametros de la red neuronal
neural_model.read_model(params["model path"],
                        params["file model"])
generate_text = generate_text_class(ngram,
                                    neural_model,
                                    tokenize)
generate_text.obtain_closet_words("pinche", 10)
generate_text.obtain_closet_words("saludo", 10)
generate_text.obtain_closet_words("twitter", 10)
