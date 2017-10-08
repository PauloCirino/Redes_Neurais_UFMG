require(tidyverse)
require(mlbench)
source('./src/methods.R')

baseImgDir <- './Article/imgs/'
if(!dir.exists(baseImgDir)){
    dir.create(baseImgDir)
}

ggplot2::ggplot(data = NULL,
                ggplot2::aes(x = X[, 1], y = X[, 2],
                             color = log(Q), z = log(Q), alpha = log(Q) )) + 
    ggplot2::geom_point(size = 2.5) + 
    ggplot2::geom_density_2d(color = 'blue') +
    ggplot2::scale_colour_continuous(low = 'green', high = 'red') +
    ggplot2::theme_bw() +
    ggplot2::labs(x = 'X1', y = 'X2') +
    ggplot2::theme(axis.title.x = element_blank(),
                   axis.text.x = element_blank(),
                   axis.ticks.x = element_blank(),
                   axis.title.y = element_blank(),
                   axis.text.y = element_blank(),
                   axis.ticks.y = element_blank(),
                   legend.position = "none") +
    ggplot2::ggsave(file = paste(baseImgDir, 'qualidade_plot.png',sep = ''))

ggplot2::ggplot(data = NULL,
                ggplot2::aes(x = Q)) +
    geom_histogram(colour = "black",
                   fill = "white",
                   bins = 30) +
    ggplot2::theme_bw() +
    ggplot2::labs(x = 'Qualidade') +
    ggplot2::theme(axis.ticks.x = element_blank(),
                   axis.title.y = element_blank(),
                   axis.text.y = element_blank(),
                   axis.ticks.y = element_blank(),
                   legend.position = "none") +
    ggplot2::ggsave(file = paste(baseImgDir, 'histograma.png',sep = ''))

ggplot2::ggplot(data = NULL,
                ggplot2::aes(x = ' ', y = Q)) +
    ggplot2::geom_boxplot() +
    ggplot2::theme_bw() +
    ggplot2::labs(x = 'Qualidade') +
    ggplot2::theme(axis.ticks.x = element_blank(),
                   axis.title.y = element_blank(),
                   axis.text.y = element_blank(),
                   axis.ticks.y = element_blank(),
                   legend.position = "none") +
    ggplot2::ggsave(file = paste(baseImgDir, 'boxplot.png',sep = ''))

ggplot2::ggplot(data = NULL,
                ggplot2::aes(x = log(Q) )) +
    geom_histogram(colour = "black",
                   fill = "white",
                   bins = 30) +
    ggplot2::theme_bw() +
    ggplot2::labs(x = 'Log(Qualidade)') +
    ggplot2::theme(axis.ticks.x = element_blank(),
                   axis.title.y = element_blank(),
                   axis.text.y = element_blank(),
                   axis.ticks.y = element_blank(),
                   legend.position = "none") +
    ggplot2::ggsave(file = paste(baseImgDir, 'histograma_log.png',sep = ''))

ggplot2::ggplot(data = NULL,
                ggplot2::aes(x = ' ', y = log(Q))) +
    ggplot2::geom_boxplot() +
    ggplot2::theme_bw() +
    ggplot2::labs(x = 'Log(Qualidade)') +
    ggplot2::theme(axis.ticks.x = element_blank(),
                   axis.title.y = element_blank(),
                   axis.text.y = element_blank(),
                   axis.ticks.y = element_blank(),
                   legend.position = "none") +
    ggplot2::ggsave(file = paste(baseImgDir, 'boxplot_log.png',sep = ''))




rm(list = ls())
require('tidyverse')
require('plotly')
require('mlbench')
require('mxnet')
require('xtable')

source('src/methods.R')
source('src/makeMNISTDataLists.R')
source('src/makePokerDataList.R')
source('src/makeSkinDataList.R')

seedsVet <- 1:10
networkSizesList <- list(c(10),
                         c(25),
                         c(50),
                         c(100),
                         c(200),
                         c(10, 5),
                         c(25, 5),
                         c(25, 10), 
                         c(50, 20), 
                         c(100, 50) )
DataList <- list()
DataList <- makeMNISTDataList(DataList = DataList)
DataList <- makePokerDataList(DataList = DataList)
DataList <- makeSkinData(DataList = DataList)

dataSetDF <- data.frame()
for(dataSetName in names(DataList)){
    Y <- DataList[[dataSetName]]['Y']
    
    nPoints <- sum(table(Y))
    dominantClassPoints <- max(table(Y))
    minorityClassPoints <- min(table(Y))
    unbalacement <- round(minorityClassPoints / dominantClassPoints, 4)
    
    dataSetDF <- rbind(dataSetDF,
                       data.frame( dataSetName = dataSetName,
                                   nPoints = nPoints,
                                   dominantClassPoints = dominantClassPoints,
                                   minorityClassPoints = minorityClassPoints,
                                   unbalacement = unbalacement)
    )
}
xtable(dataSetDF)

resultData <- readr::read_csv('./data/save_2017-10-07.csv') %>%
    dplyr::select(dataSetName, functionName, nTrainningPoints,
                  seed, networkSize, train.Accuracy, train.AUC,
                  test.Accuracy, test.AUC, compTime) %>%
    dplyr::mutate(train.Accuracy = ifelse(train.Accuracy < 0.5,
                                         1 - train.Accuracy, train.Accuracy),
                  test.Accuracy = ifelse(test.Accuracy < 0.5,
                                         1 - test.Accuracy, test.Accuracy),
                  train.AUC = ifelse(train.AUC < 0.5,
                                         1 - train.AUC, train.AUC),
                  test.AUC = ifelse(test.AUC < 0.5,
                                         1 - test.AUC, test.AUC),
                  train.Accuracy = round( 100 * train.Accuracy, 4),
                  test.Accuracy = round( 100 * test.Accuracy, 4),
                  compTime = round( 100 * compTime, 4) ) %>%
    dplyr::group_by(dataSetName, functionName, nTrainningPoints, networkSize) %>%
    dplyr::summarise(meanTrainAccuracy = mean(train.Accuracy),
                     meanTestAccuracy = mean(test.Accuracy),
                     meanTrainAUC = mean(train.AUC),
                     meanTestAUC = mean(test.AUC),
                     meanCompTime = mean(compTime) ) %>%
    dplyr::ungroup() %>%
    dplyr::group_by(dataSetName, functionName) %>%
    dplyr::top_n(n = 1, wt = meanTestAUC)
