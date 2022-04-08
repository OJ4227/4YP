# test.R

library(bnlearn)



setwd("C:/Users/order/Documents/Oxford/4th Year/4YP/R")

data = read.csv("2x2_100_samples1.csv")
labels = colnames(data)
# str(data)


data = lapply(data, as.factor)
# data[data==-1]=-1.0
# data[data==0]=0.0
# data[data==1]=1.0
# data[data==2]=2.0
# data[data==3]=3.0

# str(data)

# typeof(data)
df = as.data.frame(data)
# output = bnlearn::gs(df,test = 'x2')
# output = bnlearn::hc(df)
# output = bnlearn::mmhc(df)
output = bnlearn::hc(df)
# output
library(Rgraphviz)
graphviz.plot(output, layout = "dot")
bnlearn::amat(output)

# test_output = ci.test(x=labels[1], y=labels[7], data=df, test = 'x2')

# output2 = bnlearn::pc.stable(df)
# graphviz.plot(output2, layout = "dot")
# amat(output2)


# library(pcalg)

# data[data==-1]=4
# nlev = c(4, 4, 4, 4, 5, 5, 5, 5)
# suffStat = list(dm=df, nlev=nlev, adaptDF=FALSE)
# labels = colnames(data)
# output3 = pcalg::fci(suffStat=suffStat,indepTest = disCItest, alpha=0.05, labels=labels)
# as(output3, "amat")
# getGraph(output3)

