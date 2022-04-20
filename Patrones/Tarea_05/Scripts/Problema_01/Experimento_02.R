library(kernlab)
path_graphics <- "../../Graphics/Experiment_02_"
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
degrees <- c(1, 2, 5, 10)
for (i in 1:4) {
    file_graphics <- paste(path_graphics,
        as.character(i),
        ".pdf",
        sep = ""
    )
    pdf(file = file_graphics)
    s <- ksvm(categoria ~ X + Y,
        data = d,
        kernel = "p",
        C = 11,
        kpar = list(
            degree = degrees[i],
            offset = 0
        )
    )
    plot(s, data = d)
    title(main = paste("                                      ",
        "degree =",
        as.character(degrees[i]),
        sep = " "
    ))
    dev.off()
}