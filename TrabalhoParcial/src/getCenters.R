getCenters <- function(X, Y){
    M <- ncol(X)
    Ylevels <- unique(Y)
    nClasses <- length(Ylevels)
    
    centers <- matrix(nrow = nClasses, ncol = M)
    for(i in 1:nClasses ){
        y <- Ylevels[i]
        pos <- y == Y
        
        auxX <- X[pos, ]
        iter_center <- apply(auxX, 2, mean)
        centers[i, ] <- iter_center
    }
    
    centers
}
