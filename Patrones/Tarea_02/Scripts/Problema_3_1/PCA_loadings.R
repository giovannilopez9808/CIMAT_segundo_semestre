source("Functions.R")
path_data <- "../../Data/mnist_test.csv"
# Lectura de los datos
data <- read.csv(path_data)
# Tomar unicamente la clase 0
data <- data[data$label == 0, ]
# Eliminacion de la columna label
data <- data[, -1]
data <- apply(data, 1, rescale)
pca <- prcomp(data)
loadings <- pca$x
loadings <- normalize_image(loadings)
write.csv(
    loadings,
    "../../Results/Problema_3_1/loadings.csv",
    quote = FALSE
)