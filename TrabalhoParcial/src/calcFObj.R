calcFObj <- function(Data, U, centers, expo, distanceMetric){
    mf <- U ** expo
    D <- flexclust::dist2(x = centers, y = Data, method = distanceMetric)
    sum( (D ** 2) * mf ) 
}