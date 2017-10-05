source('./src/estimateU.R')
stepBrocador <- function(Data, U, K, expo, distanceMetric = distanceMetric){
    MF <- U ** expo
    sumMF <- apply(MF, 1, sum)
    centers <- apply( (MF %*% Data), 2, function(x) {x / sumMF} )
    U <- estimateU(Data = Data, 
                   centers = centers, 
                   expo = expo,
                   distanceMetric = distanceMetric)
    
    list(U = U,
         centers = centers)
}