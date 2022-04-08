# test1


library(pcalg)
list_of_dims = c()
x = c(1:2)
for (i in x) {
  data = read.csv("2x2_50000_samples1.csv")
  labels = colnames(data)
  
  df = as.data.frame(data)
  score = new("GaussL0penObsScore", df)
  num_grids = length(labels)/2
  nlev = c(rep(4,num_grids),rep(5,num_grids))
  suffStat = list(dm=df, nlev=nlev, adaptDF=FALSE)
  
  
}

data = read.csv("2x2_50000_samples1.csv")
labels = colnames(data)

df = as.data.frame(data)
score = new("GaussL0penObsScore", df)
nlev = c(4, 4, 4, 4, 5, 5, 5, 5)
suffStat = list(dm=df, nlev=nlev, adaptDF=FALSE)
# labels = colnames(data)


fci1 = algorithm1(fci())
output3 = fci1(suffStat=suffStat,indepTest = disCItest, alpha=0.05, labels=labels)

adj_matrix = as(output3, "amat")

plot(output3)
adj_matrix