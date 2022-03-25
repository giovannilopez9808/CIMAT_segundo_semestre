from Modules.models import Mex_data_class, model_class, neural_language_model, generate_text_class
from Modules.datasets import get_args, get_params, init_seeds
from nltk.tokenize import TweetTokenizer as tokenizer
from Modules.ngrams_class import ngram_model


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
neural_model.read_model(params["path data"],
                        params["file model"])
generate_text = generate_text_class(ngram,
                                    neural_model,
                                    tokenize)
generate_text.run("<s> <s> <s>")
