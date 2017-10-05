####
# A Diferença entre briFCM e briFCM2 é o que o briFCM2 recebe os Centros iniciais,
# por outro lado briFCM recebe o valor K
####

source('./src/initU.R')
source('./src/calcFObj.R')
source('./src/stepBrocador.R')

briFCM2 <- function(Data, K, centers,
                    expo = 2,
                    removePerc = 0.1,
                    nIter = 100, 
                    minFObjt = 0.001, 
                    distanceMetric = 'euclidean',
                    verbose = TRUE,
                    verboseGraphs = FALSE,
                    waitTime = 2,
                    orderRemove = 1){
    originalData <- Data
    Data <- as.matrix(Data)
    
    nObs <- nrow(Data)
    nVars <- ncol(Data)
    
    maxRemovePerc <- 1 - removePerc
    minNumPoints <- round( (1 - maxRemovePerc) * nObs)
    
    initTS <- Sys.time()
    initUResult <- initU( Data = Data, 
                          centers = centers,
                          expo = expo,
                          distanceMetric = distanceMetric,
                          verboseGraphs = verboseGraphs,
                          waitTime = waitTime)
    
    centers <- initUResult$centers
    U <- initUResult$U
    
    lastFObj <- Inf
    FObjt <- calcFObj(Data = Data,
                      U = U,
                      centers = centers, 
                      expo = expo, 
                      distanceMetric = distanceMetric)
    
    i <- 0
    while( (i < nIter) && (nrow(Data) > minNumPoints) && ( lastFObj - FObjt ) > minFObjt ){
        i <- i + 1
        
        fcmStepResult <- stepBrocador(Data = Data,
                                      U = U,
                                      K = K,
                                      expo = expo,
                                      distanceMetric = distanceMetric)
        
        U <- fcmStepResult$U
        centers <- fcmStepResult$centers
        
        Y <- apply(U, 2, which.max)
        Q <- apply(U, 2, function(x){ 1 - (K**K) * prod(1 / x)})
        
        if(verboseGraphs && (waitTime > 0 ) ){
            p <- plotCenterAndData(Data = Data, Y = Y, centers = centers,
                                   subTitle = paste('Iter -', i))
            print(p)
            Sys.sleep(waitTime)
        }
        
        if( removePerc > 0 && removePerc < 1){
            
            nKeepObs <- round( (1 - removePerc) * nrow(Data) )
            pos <- order(orderRemove*Q)[1:nKeepObs]
            
            centersRemovedCount <- as.numeric( table( Y[-pos] ) )
            
            U <- U[, pos]
            Data <- Data[pos, ]
            
        }
        
        lastFObj <- FObjt
        FObjt <- calcFObj(Data = Data,
                          U = U,
                          centers = centers, 
                          expo = expo, 
                          distanceMetric = distanceMetric)
        
    }
    
    lastU <- estimateU(Data = originalData, 
                       centers = centers, 
                       expo = expo,
                       distanceMetric = distanceMetric)
    
    Y <- apply(lastU, 2, which.max)
    
    list(U = U,
         centers = centers,
         Y = Y,
         nIter = i,
         initTS = initTS,
         endTS = Sys.time())
}