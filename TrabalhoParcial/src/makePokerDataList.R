source('./src/methods.R')
makePokerDataList <- function(DataList = list()){
    Data <- dplyr::bind_rows(readr::read_csv("data/poker-hand-training-true.data", col_names = FALSE),
                                  readr::read_csv("data/poker-hand-testing.data", col_names = FALSE))
    X <- data.matrix(Data)[, 1:10]
    X <- apply(X = X, MARGIN = 2, FUN = function(x){
        x / max(X)
    }) 
    
    Y <- numeric()
    Y[!(Data$X11 >= 4)] <- 0
    Y[Data$X11 >= 4] <- 1
    
    Q <- getQuality(X = X, Y = Y, expo = 2, distanceMetric = 'euclidean')
    DataList[['Poker Data']] <- list(X = X,
                                     Y = Y,
                                     Q = Q)
    DataList
}