library(kernlab)
library(mlbench)
path_graphics <- "../../Graphics/Experiment_04_"
ds <- mlbench.spirals(300, 2, 0.03)
sigmas <- c(0.5, 5, 10, 20)
for (i in 1:4) {
    file_graphics <- paste(path_graphics,
        as.character(i),
        ".pdf",
        sep = ""
    )
    pdf(file = file_graphics)
    s <- ksvm(classes ~ .,
        data = ds,
        kernel = "r",
        C = 1,
        kpar = list(sigma = 1 / sigmas[i])
    )
    plot(s, data = ds)
    title(main = paste("                                      ",
        "sigma =",
        as.character(sigmas[i]),
        sep = " "
    ))
    dev.off()
}