from Modules.models import Mex_data_class, ngram_model, neural_language_model, model_class, generate_text_class, log_likelihood, syntax_structure, perplexity
from Modules.datasets import init_seeds, get_args, get_params
from nltk.tokenize import TweetTokenizer as tokenizer

init_seeds()
params = get_params()
args = get_args()
tokenize = tokenizer().tokenize
print("Lectura de archivos")
mex_data = Mex_data_class(params, args)
ngram = ngram_model(args.N, tokenize=tokenize)
ngram.fit(mex_data.train_data)
args.vocabulary_size = ngram.get_vocabulary_size()
mex_data.obtain_data_and_labels(ngram)
mex_data.obtain_loaders()
print("Init neural model")
neural_model = neural_language_model(args)
model = model_class(neural_model,
                    args,
                    mex_data.train_loader,
                    mex_data.validation_loader)
print("Train neural model")
# model.run()
neural_model.read_model(params["model path"],
                        params["file model"])
generate_text = generate_text_class(ngram,
                                    neural_model,
                                    tokenize)
print("-"*40)
print("Primer palabra")
print(generate_text.run("hello"))
print("-"*40)
print("Segunda palabra")
print(generate_text.run("corre"))
print("-"*40)
print("Tercera palabra")
print(generate_text.run("<s> <s> c"))

print("log likelihood", log_likelihood(neural_model,
                                       "Dejalo que termine",
                                       ngram))
print("log likelihood", log_likelihood(neural_model,
                                       "esperate a que tenga servicios, ya completos",
                                       ngram))
print("log likelihood", log_likelihood(neural_model,
                                       "asi te ganas un chingo de gente",
                                       ngram))
print("log likelihood", log_likelihood(neural_model,
                                       "eso que esten en redes con sus criticas",
                                       ngram))
print("log likelihood", log_likelihood(neural_model,
                                       "unas tlayudas no le hacen daño a nadie",
                                       ngram))
word = "enojada"
syntax_structure(neural_model, ngram, word)
perplexity(neural_model,
           mex_data.validation_text,
           ngram)
