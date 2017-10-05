rm(list = ls())
require('tidyverse')
require('plotly')
require('mlbench')
source('src/methods.R')
require('mnist')

data(mnist) 
X <- rbind(mnist$train$x, mnist$test$x)
Y <- c(mnist$train$y, mnist$test$y)

pos5 <- Y == 5
Y[pos5] <- 1
Y[!pos5] <- 0
    
Q <- getQuality(X = X, Y = Y)
networkSize <- c(60, 10)
seed <- 1234
methodName <- 'briSelectionPlusPlusLogNeg'

iterResult <- runModel(X = X, Y = Y, Q = Q, 
                       networkSize = networkSize,
                       functionName = methodName,
                       seed = seed)
