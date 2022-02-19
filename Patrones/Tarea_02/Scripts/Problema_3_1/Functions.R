if (!require(scales)) {
    install.packages("scales", dependencies = TRUE)
    library(scales)
}
normalize_image <- function(data) {
    data_normalize <- apply(
        data,
        1,
        rescale
    )
    return(data_normalize)
}
rotate <- function(x) {
    t(apply(x, 2, rev))
}

plot_image <- function(image) {
    image(rotate(rotate(image)),
        col = grey(seq(0, 1, length = 256))
    )
}

reshape_image <- function(vector) {
    image <- matrix(t(vector),
        ncol = 28
    )
    return(image)
}