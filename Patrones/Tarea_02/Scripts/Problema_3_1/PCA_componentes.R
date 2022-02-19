source("Functions.R")
path_data <- "../../Data/mnist_test.csv"
path_results <- "../../Results/Problema_3_1/pca_components.csv"
# Lectura de los datos
data <- read.csv(path_data)
# Tomar unicamente la clase 0
data <- data[data$label == 0, ]
# Eliminacion de la columna label
data <- data[, -1]
data <- normalize_image(data)
pca <- prcomp(t(data))
positions <- data.frame(pca$rotation)
positions <- normalize_image(positions)
write.csv(
    positions,
    path_results,
    quote = FALSE
)