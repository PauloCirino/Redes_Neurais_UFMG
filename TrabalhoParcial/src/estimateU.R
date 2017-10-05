estimateU <- function(Data, centers, expo, distanceMetric){
    D <- flexclust::dist2(x = Data, y = centers, method = distanceMetric)
    aux <- D ** (-2 / (expo - 1))
    auxSum <- apply(aux, 2, sum)
    
    U <- t( apply( aux, 1, function(x) {x / auxSum} ) )
    U[is.na(U)] <- 1
    
    U
}