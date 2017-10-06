require('tidyverse')
require('mxnet')
require('pROC')
require('caret')

source('./src/getCenters.R')
source('./src/estimateU.R')

getQuality <- function(X, Y, expo = 2, distanceMetric = 'euclidean'){
    K <- length(unique(Y))
    centers <- getCenters(X = X, Y = Y)
    U <- estimateU(Data = X,
                   centers = centers, 
                   expo = expo, 
                   distanceMetric = distanceMetric)
    Q <- -1 * ( apply(U, 1, function(x){ 1 - (K**K) * prod(1 / x)}) )
    Q
}

baseLine <- function(X, Y, Q, seed = Sys.time()){
    list(X = X %>% data.matrix(),
         Y = Y,
         Q = NA ) 
}

briSelection <- function(X, Y, Q, seed = Sys.time()){
    set.seed(seed)
    
    Data <- data.frame(X = X, Y = Y, Q = Q)
    
    auxMinrityClass <- Data %>% 
        dplyr::group_by(Y) %>%
        dplyr::summarise(count = n()) %>%
        dplyr::top_n(n = -1, wt = count)
    
    minorityY <- auxMinrityClass$Y
    minorityCount <- auxMinrityClass$count
    
    Data <- Data %>%
        dplyr::group_by(Y) %>%
        dplyr::top_n(n = minorityCount, wt = -Q) %>%
        dplyr::ungroup() %>%
        data.frame()
    
    list(X = Data %>% dplyr::select(-Q, -Y) %>% data.matrix(),
         Y = Data$Y,
         Q = Data$Q
    )
}

briSelectionPlusPlus <- function(X, Y, Q, seed = Sys.time()){
    set.seed(seed)
    
    Data <- data.frame(X = X, Y = Y, Q = Q)
    
    auxMinrityClass <- Data %>% 
        dplyr::group_by(Y) %>%
        dplyr::summarise(count = n()) %>%
        dplyr::top_n(n = -1, wt = count)
    
    minorityY <- auxMinrityClass$Y
    minorityCount <- auxMinrityClass$count
    
    Data <- Data %>%
        dplyr::group_by(Y) %>%
        dplyr::sample_n(size = minorityCount, replace = FALSE, weight = Q) %>%
        dplyr::ungroup() %>%
        data.frame()
    
    list(X = Data %>% dplyr::select(-Q, -Y) %>% data.matrix(),
         Y = Data$Y,
         Q = Data$Q
    )
}

briSelectionPlusPlusNeg <- function(X, Y, Q, seed = Sys.time()){
    set.seed(seed)
    
    Data <- data.frame(X = X, Y = Y, Q = Q)
    
    auxMinrityClass <- Data %>% 
        dplyr::group_by(Y) %>%
        dplyr::summarise(count = n()) %>%
        dplyr::top_n(n = -1, wt = count)
    
    minorityY <- auxMinrityClass$Y
    minorityCount <- auxMinrityClass$count
    
    Data <- Data %>%
        dplyr::mutate(Q = 1/Q) %>%
        dplyr::group_by(Y) %>%
        dplyr::sample_n(size = minorityCount, replace = FALSE, weight = Q) %>%
        dplyr::ungroup() %>%
        data.frame()
    
    list(X = Data %>% dplyr::select(-Q, -Y) %>% data.matrix(),
         Y = Data$Y,
         Q = Data$Q
    )
}

briSelectionPlusPlusLog <- function(X, Y, Q, seed = Sys.time()){
    set.seed(seed)
    
    Data <- data.frame(X = X, Y = Y, Q = Q)
    
    auxMinrityClass <- Data %>% 
        dplyr::group_by(Y) %>%
        dplyr::summarise(count = n()) %>%
        dplyr::top_n(n = -1, wt = count)
    
    minorityY <- auxMinrityClass$Y
    minorityCount <- auxMinrityClass$count
    
    Data <- Data %>%
        dplyr::mutate(Q = log(Q)) %>%
        dplyr::group_by(Y) %>%
        dplyr::sample_n(size = minorityCount, replace = FALSE, weight = Q) %>%
        dplyr::ungroup() %>%
        data.frame()
    
    list(X = Data %>% dplyr::select(-Q, -Y) %>% data.matrix(),
         Y = Data$Y,
         Q = Data$Q
    )
}


briSelectionPlusPlusLogNeg <- function(X, Y, Q, seed = Sys.time()){
    set.seed(seed)
    
    Data <- data.frame(X = X, Y = Y, Q = Q)
    
    auxMinrityClass <- Data %>% 
        dplyr::group_by(Y) %>%
        dplyr::summarise(count = n()) %>%
        dplyr::top_n(n = -1, wt = count)
    
    minorityY <- auxMinrityClass$Y
    minorityCount <- auxMinrityClass$count
    
    Data <- Data %>%
        dplyr::mutate(Q = 1/log(Q)) %>%
        dplyr::group_by(Y) %>%
        dplyr::sample_n(size = minorityCount, replace = FALSE, weight = Q) %>%
        dplyr::ungroup() %>%
        data.frame()
    
    list(X = Data %>% dplyr::select(-Q, -Y) %>% data.matrix(),
         Y = Data$Y,
         Q = Data$Q
    )
}

runModel <- function(X, Y, Q, networkSize, functionName,
                     seed = Sys.time(),
                     learnRate = 0.05, momentum = 0.01, 
                     maxit = 100, evalMetric = mxnet::mx.metric.accuracy,
                     trainAndTestRatio = 0.3){
    set.seed(seed)
    method <- get(functionName)
    
    pos <- sample(length(Y), round((1-trainAndTestRatio) * length(Y)))
    
    Xtrain <- X[pos, ]
    Xtest <- X[-pos, ]
    Ytrain <- Y[pos]
    Ytest <- Y[-pos]
    Qtrain <- Q[pos]
    Qtest <- Q[-pos]
    
    result <- method(X = Xtrain, Y = Ytrain, Q = Qtrain, seed = seed)
    XtrainSelected <- as.matrix( result[['X']] )
    YtrainSelected <- as.numeric( result[['Y']] )
    QtrainSelected <- result[['Q']]
    colnames(XtrainSelected) <- colnames(X)
    
    model <- mxnet::mx.mlp( XtrainSelected, YtrainSelected,
                            hidden_node = networkSize,
                            out_node = 2,
                            out_activation = "softmax",
                            num.round = maxit, 
                            learning.rate = learnRate,
                            momentum = momentum,
                            eval.metric = evalMetric )
    
    predYtrain <- t( predict(model, XtrainSelected) )
    predYtest <- t( predict(model, Xtest) )
    
    trainConfMatrix <- caret::confusionMatrix(round(predYtrain[, 2]), YtrainSelected)
    trainAUC <- ModelMetrics::auc(actual = YtrainSelected, predicted = predYtrain[, 2])
    trainMetrics <- c(trainConfMatrix$overall, trainConfMatrix$byClass, AUC = trainAUC)
    
    testConfMatrix <- caret::confusionMatrix(round(predYtest[, 2]), Ytest)
    testAUC <- ModelMetrics::auc(actual = Ytest, predicted = predYtest[, 2])
    testMetrics <- c(testConfMatrix$overall, testConfMatrix$byClass, AUC = testAUC)
    
    results <- c(functionName = functionName,
                 seed = seed,
                 trainAndTestRatio = trainAndTestRatio,
                 maxit = maxit,
                 learnRate = learnRate, momentum = momentum,
                 networkSize = paste(networkSize, collapse = '-'),
                 train = trainMetrics, test = testMetrics)
    
}


runAllTests <- function(DataList,
                        networkSizesList,
                        seedsVet,
                        methodsNames = c('baseLine', 'briSelection',
                                         'briSelectionPlusPlus', 
                                         'briSelectionPlusPlusNeg', 
                                         'briSelectionPlusPlusLog',
                                         'briSelectionPlusPlusLogNeg'),
                        saveFile = paste('./data/save_', Sys.Date(), '.csv', sep = ''),
                        echoEachNMin = 10){
    itersTable <- expand.grid(dataSetName = names(DataList),
                              netWorkSizePos = 1:length(networkSizesList),
                              seed = seedsVet,
                              methodName = methodsNames) %>%
        dplyr::arrange(seed, netWorkSizePos, dataSetName, methodName)
    
    globalResult <- data.frame()
    lastSaveTS <- as.numeric(Sys.time())
    NTotalIters <- nrow(itersTable)
    for(i in 1:NTotalIters){
        iterData <- itersTable[i, ]
        
        DataSet <- DataList[[iterData$dataSetName]]
        iterX <- DataSet$X
        iterY <- DataSet$Y
        iterQ <- DataSet$Q
        
        networkSize <- networkSizesList[[iterData$netWorkSizePos]]
        seed <- iterData$seed
        methodName <- as.character( iterData$methodName )
        
        iterResult <- runModel(X = iterX, Y = iterY, Q = iterQ, 
                               networkSize = networkSize,
                               functionName = methodName,
                               seed = seed)
        
        auxResults <- data.frame( t(c(dataSetName = iterData$dataSetName, 
                                      iterResult) ) )
        globalResult <- dplyr::bind_rows(globalResult, auxResults)
        
        if(as.numeric(Sys.time()) - lastSaveTS > 60 * echoEachNMin){
            cat('Iter ', i, '/', NTotalIters, ' - At', as.character(Sys.time()))
            write.csv(x = globalResult,
                      file = saveFile  )
        }
    }
    write.csv
}