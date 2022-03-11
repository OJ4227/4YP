# pcalg algorithms (FCI, GES, GIES)

setwd("C:/Users/order/Documents/Oxford/4th Year/4YP/R")
library(pcalg)

data = read.csv("2x2_50000_samples1.csv")
labels = colnames(data)

results = data.frame()

df = as.data.frame(data)
score = new("GaussL0penObsScore", df)
nlev = c(4, 4, 4, 4, 5, 5, 5, 5)
suffStat = list(dm=df, nlev=nlev, adaptDF=FALSE)
labels = colnames(data)
output3 = fci(suffStat=suffStat,indepTest = disCItest, alpha=0.05, labels=labels)

adj_matrix = as(output3, "amat")

plot(output3)
adj_matrix
# write.csv(adj_matrix, "C:\\Users\\order\\Documents\\Oxford\\4th Year\\4YP\\Pyro Practice\\adj.csv")
