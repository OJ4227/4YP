# pcalg test

#library(pcalg)
#data("gmG")

#suffStat = list(C = cor(gmG8$x), n = nrow(gmG8$x))
#varNames = gmG8$g@nodes
#skel.gmG8 = skeleton(suffStat, indepTest = gaussCItest, labels = varNames, alpha = 0.01)
#pc.gmG8 <- pc(suffStat, indepTest = gaussCItest, labels = varNames, alpha = 0.01)
#suffStat

library(pcalg)

## Simulate data
# n <- 100
# set.seed(123)
# x <- sample(0:2, n, TRUE) ## three levels
# y <- sample(0:3, n, TRUE) ## four levels
# z <- sample(0:1, n, TRUE) ## two levels
# dat <- cbind(x,y,z)
# 
# ## Analyze data
# gSquareDis(1,3, S=2, dat, nlev = c(3,4,2)) # but nlev is optional:
# gSquareDis(1,3, S=2, dat, verbose=TRUE, adaptDF=TRUE)
# ## with too little data, gives a warning (and p-value 1):
# gSquareDis(1,3, S=2, dat[1:60,], nlev = c(3,4,2))
# 
# suffStat <- list(dm = dat, nlev = c(3,4,2), adaptDF = FALSE)
# disCItest(1,3,2,suffStat = suffStat)

data("gmD")
gmD
labels = colnames(gmD$x)
suffStat = list(dm=gmD$x, nlev = c(3,2,3,4,2), adaptDF=FALSE)
p = nrow(gmD$x)
output4 = pcalg::pc(suffStat, indepTest = disCItest, alpha = 0.05, labels = labels)

                    