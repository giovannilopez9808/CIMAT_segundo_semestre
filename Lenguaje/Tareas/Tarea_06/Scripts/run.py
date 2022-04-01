from Modules.nn_models import CNNTextCls, model_class, save_stadistics
from Modules.datasets import get_args, get_params
from Modules.Mex_data import Mex_data_class

params = get_params()
args = get_args()
data = Mex_data_class(params, args)
CNN_model = CNNTextCls(args)
model = model_class(CNN_model,
                    args,
                    data.train_loader,
                    data.validation_loader)
stadistics = model.run()
save_stadistics(params,
                stadistics)
