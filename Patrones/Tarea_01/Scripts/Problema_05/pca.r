library(stringr)
data <- read.csv("../../Data/heptatlon.csv", row.names = 1)
names <- rownames(data)
names <- word(names, 1)
names <- str_to_title(names)
rownames(data) <- names
data_pca <- data[, -1]
pca <- princomp(data_pca,
    cor = T
)
print(summary(pca))
print(pca$loadings)
biplot(pca$loadings)