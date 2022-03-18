from Modules.dataset import obtain_parms, image_class

params = obtain_parms()
image = image_class(params)
image.run_kmeans()
image.compress()
image.save()
