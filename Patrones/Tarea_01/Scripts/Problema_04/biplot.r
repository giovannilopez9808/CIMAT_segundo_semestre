data <- matrix(scan("../../Data/oef2.dat"),
    35,
    12,
    byrow = T
)
nombresestaciones <- c(
    "St. John_s",
    "Charlottetown",
    "Halifax",
    "Sydney",
    "Yarmouth",
    "Fredericton",
    "Arvida",
    "Montreal",
    "Quebec City",
    "Schefferville",
    "Sherbrooke",
    "Kapuskasing",
    "London",
    "Ottawa",
    "Thunder Bay",
    "Toronto",
    "Churchill",
    "The Pas",
    "Winnipeg",
    "Prince Albert",
    "Regina",
    "Beaverlodge",
    "Calgary",
    "Edmonton",
    "Kamloops",
    "Prince George",
    "Prince Rupert",
    "Vancouver",
    "Victoria",
    "Dawson",
    "Whitehorse",
    "Frobisher Bay",
    "Inuvik",
    "Resolute",
    "Yellowknife"
)
rownames(data) <- nombresestaciones
pca <- princomp(data,
    cor = T
)
components <- pca$loadings
biplot(pca)