from Modules.results_model import results_model
from Modules.image_model import image_model
from Modules.params import get_params


params = get_params()
results = results_model(params)
image = image_model(params)
for image_name in params["Images"]:
    results.read(image_name)
    image.segmentation(results.image,
                       3,
                       results.alpha,
                       results.mu,
                       0.5)
    image.histogram_segmentation(results.image,
                                 3,
                                 results.h_0,
                                 results.h_1)
    image.plot(results.image,
               image_name)
