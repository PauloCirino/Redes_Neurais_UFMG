require(tidyverse)
require(mlbench)
source('./src/methods.R')

baseImgDir <- './Article/imgs'
if(!dir.exists(baseImgDir)){
    dir.create(baseImgDir)
}

Data <- mlbench::mlbench.2dnormals(n = 500, sd = 0.55)
X <- Data$x
Y <- Data$classes
Q <- getQuality(X = X, Y = as.numeric(Y) - 1)
Data <- data.frame(X1 = X[, 1], X2 = X[, 2], Y = as.numeric(Y) - 1, Q = Q) 

Data %>%
    ggplot2::ggplot(ggplot2::aes(x = X1, y = X2, color = Q)) +
    ggplot2::geom_point() + 
    ggplot2::scale_color_continuous()

