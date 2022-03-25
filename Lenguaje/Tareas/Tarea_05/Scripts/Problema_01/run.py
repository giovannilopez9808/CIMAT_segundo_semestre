from Modules.datasets import init_seeds, get_args, get_params
from Modules.models import Mex_data_class, ngram_model, neural_language_model, model_class, generate_text_class
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
print(generate_text.run("pende"))
