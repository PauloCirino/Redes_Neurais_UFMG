require('mnist')
source('./src/methods.R')

makeMNISTDataList <- function(DataList = list()){
    mnist <- download_mnist()
    X <- as.matrix( mnist[, -785] / 255 )
    Y <- as.numeric( as.character( mnist[, 785] ) )
    
    for(i in unique(Y)){
        iterY <- numeric()
        
        iterPos <- Y == i
        iterY[iterPos] <- 1
        iterY[!iterPos] <- 0
        
        iterQ <- getQuality(X = X, Y = iterY, expo = 2, distanceMetric = 'euclidean')
        iterName <- paste(i, ' vs ALL', sep = '' )
        
        DataList[[iterName]] <- list(X = X,
                                     Y = iterY,
                                     Q = iterQ)
    }   
    DataList
}