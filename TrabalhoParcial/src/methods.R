require('tidyverse')
require('RSNNS')
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

runModel <- function(X, Y, Q, networkSize, functionName, expo = 2, distanceMetric = 'euclidean',
                     seed = Sys.time(),
                     learnFuncParams = 0.1, maxit = 5000, trainAndTestRatio = 0.3){
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
    XtrainSelected <- result[['X']]
    YtrainSelected <- result[['Y']]
    QtrainSelected <- result[['Q']]
    
    
    YtrainSelected <- RSNNS::decodeClassLabels( x = YtrainSelected)
    Ytest <- RSNNS::decodeClassLabels( x = Ytest )
    
    model <- RSNNS::mlp(x = XtrainSelected, y = YtrainSelected,
                        size = networkSize, learnFuncParams = learnFuncParams, 
                        maxit = maxit, 
                        inputsTest = Xtest, targetsTest = Ytest)
    
    predYtrain <- model$fitted.values
    predYtest <- model$fittedTestValues
    
    trainConfMatrix <- caret::confusionMatrix(round(predYtrain[, 1]), YtrainSelected[, 1])
    trainAUC <- ModelMetrics::auc(actual = YtrainSelected[, 1], predicted = predYtrain[, 1])
    trainMetrics <- c(trainConfMatrix$overall, trainConfMatrix$byClass, AUC = trainAUC)
    
    testConfMatrix <- caret::confusionMatrix(round(predYtest[, 1]), Ytest[, 1])
    testAUC <- ModelMetrics::auc(actual = Ytest[, 1], predicted = predYtest[, 1])
    testMetrics <- c(testConfMatrix$overall, testConfMatrix$byClass, AUC = testAUC)
    
    results <- c(functionName = functionName,
                 seed = seed,
                 trainAndTestRatio = trainAndTestRatio,
                 maxit = maxit,
                 learnFuncParams = learnFuncParams,
                 expo = expo,
                 distanceMetric = distanceMetric,
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
                        saveFile = paste('save_', Sys.Date(), '.csv', sep = ''),
                        echoEachNMin = 1){
    itersTable <- expand.grid(dataSetName = names(DataList),
                              netWorkSizePos = 1:length(networkSizesList),
                              seed = seedsVet,
                              methodName = methodsNames)
    globalResult <- data.frame()
    lastSaveTS <- as.numeric(Sys.time())
    for(i in 1:nrow(itersTable)){
        iterData <- itersTable[i, ]
        
        DataSet <- DataList[[iterData$dataSetName]]
        iterX <- DataSet$X
        iterY <- DataSet$Y
        iterQ <- getQuality(X = iterX, Y = iterY)
        
        networkSize <- networkSizesList[[iterData$netWorkSizePos]]
        seed <- iterData$seed
        methodName <- iterData$methodName
        
        iterResult <- runModel(X = iterX, Y = iterY, Q = iterQ, 
                               networkSize = networkSize,
                               functionName = methodName,
                               seed = seed)
        globalResult[i, ] <- c( dataSetName = iterData$dataSetName,
                                iterResult)
        write.csv(x = c( dataSetName = iterData$dataSetName, iterResult),
                  file = saveFile,
                  append = TRUE)
        
        if(as.numeric(Sys.time()) - lastSaveTS > 60 * saveEachNMin){
            cat('Iter ', i, '- At', as.character(Sys.time()))
            print(globalResult)
            
        }
    }
}