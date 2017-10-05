####
# A Diferença entre initU e initU2 é o que o initU2 recebe os Centros iniciais,
# por outro lado initU recebe o número de valores a serem inicializados
####

require('flexclust')

initU <- function(Data, centers, expo, distanceMetric, verboseGraphs, waitTime){
    U <- estimateU(Data = Data,
                   centers = centers, 
                   expo = expo, 
                   distanceMetric = distanceMetric)
    
    Y <- apply(U, 2, which.max)
    
    list(centers = centers,
         U = U,
         Y = Y)
}