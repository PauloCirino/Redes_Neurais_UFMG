rm(list = ls())
require('tidyverse')
require('plotly')
require('mlbench')
require('mxnet')

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


results <- runAllTests(DataList = DataList, 
                        networkSizesList = networkSizesList,
                        seedsVet = seedsVet)
