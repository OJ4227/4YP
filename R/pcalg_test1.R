library(pcalg)

setwd("C:/Users/order/Documents/Oxford/4th Year/4YP/R")

data = read.csv("3x3_5000_samples1.csv")
labels = colnames(data)
# str(data)

# data[data==-1]=4
df = as.data.frame(data)
# nlev = c(4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5)
# suffStat = list(dm=df, nlev=nlev, adaptDF=FALSE)
# labels = colnames(data)
# output3 = fci(suffStat=suffStat,indepTest = disCItest, alpha=0.05, labels=labels)

score = new("GaussL0penObsScore", df)
output3 = ges(score)
# as(output3, "amat")
# results = data.frame()

# library(Rgraphviz)

plot(output3$essgraph)


