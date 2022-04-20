library(kernlab)
path_graphics <- "../../Graphics/Experiment_03_"
x1 <- rnorm(30, 0, 1)
y1 <- rnorm(30, 0, 1)
x2 <- rnorm(30, 2, 1)
y2 <- rnorm(30, 2, 1)
c <- c(
    rep(0, 30),
    rep(1, 30)
)
categoria <- factor(c)
d <- data.frame(
    c(x1, x2),
    c(y1, y2),
    categoria
)
names(d) <- c("X", "Y", "categoria")
s <- ksvm(categoria ~ X + Y,
    data = d,
    kernel = "p",
    cost = 1,
    kpar = list(
        degree = 1,
        offset = 0
    )
)
x1 <- rnorm(30, 0, 1)
y1 <- rnorm(30, 0, 1)
x2 <- rnorm(30, 2, 1)
y2 <- rnorm(30, 2, 1)
c <- c(
    rep(0, 30),
    rep(1, 30)
)
categoria <- factor(c)
dtest <- data.frame(
    c(x1, x2),
    c(y1, y2),
    categoria
)
names(dtest) <- c(
    "X",
    "Y",
    "categoria"
)
test <- table(
    pred = predict(s, dtest),
    truth = dtest$categoria
)
print(test)
sigmas <- c(0.5, 5, 10, 20)
for (i in 1:4) {
    file_graphics <- paste(path_graphics,
        as.character(i),
        ".pdf",
        sep = ""
    )
    pdf(file = file_graphics)
    s <- ksvm(categoria ~ X + Y,
        data = d,
        kernel = "r",
        C = 1,
        kpar = list(
            sigma = 1 / sigmas[i]
        )
    )
    plot(s, data = d)
    title(main = paste("                                      ",
        "sigma =",
        as.character(sigmas[i]),
        sep = " "
    ))
    dev.off()
}