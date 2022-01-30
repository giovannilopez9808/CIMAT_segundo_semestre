library(ggplot2)
library(reshape2)
data <- matrix(scan("../Data/oef2.dat"),
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
components <- data.frame(
    x = seq(1, 12),
    y1 = components[, 1],
    y2 = components[, 2]
)
dm <- melt(components,
    id.var = 1
)
ggplot(
    data = dm,
    aes(x,
        value,
        colour = variable
    )
) +
    geom_point() +
    xlim(0, 13) +
    ylim(-0.5, 0.5) +
    scale_colour_manual(
        labels = c("1ra", "2da"),
        values = c("#6a040f", "#e85d04")
    ) +
    labs(x = "", y = "Componente", color = "Componentes") +
    theme_bw()
ggsave("../Document/Graphics/components.png",
    height = 8,
    width = 16,
    units = "cm"
)