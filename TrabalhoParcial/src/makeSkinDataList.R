source('./src/methods.R')
makeSkinData <- function(DataList = list()){
    Data <- dplyr::bind_rows(readr::read_tsv("data/Skin_NonSkin.txt", col_names = FALSE)) %>%
        data.frame()
    
    X <- data.matrix(Data)[, 1:3]
    X <- apply(X = X, MARGIN = 2, FUN = function(x){
        x / max(X)
    }) 
    
    Y <- as.numeric(Data[, 4]) - 1
    
    Q <- getQuality(X = X, Y = Y, expo = 2, distanceMetric = 'euclidean')
    DataList[['Skin Data']] <- list(X = X,
                                     Y = Y,
                                     Q = Q)
    DataList
}