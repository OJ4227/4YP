# pcalg algorithms (FCI)
rm(list=ls())
setwd("C:/Users/order/Documents/Oxford/4th Year/4YP/R")
library(pcalg)
library(stringr)

amat_to_dataframe = function(adj_matrix,labels,dims) {
  len = dims[1]*dims[2]*2
  
  df1 = data.frame(matrix(ncol = len, nrow = len))
  colnames(df1) = labels
  
  elements = c(1:len)
  for (i in elements) {
    start = (i-1)*len + 1
    end = i*len
    df1[i,] = adj_matrix[start:end]
  }
  return(df1)
}


sig_levels = c(0.01,0.05,0.1)

list_of_dims = list(c(2,2),c(3,3),c(4,4),c(2,4),c(4,2))
num_samples = c(100,200,500,1000,2000,5000,10000,20000,50000)
x = c(1:10)


for (dims in list_of_dims) {
  for (num in num_samples) {
    for (i in x) {
      
      # Load in and format data
      data_path = str_glue("C:\\Users\\order\\Documents\\Oxford\\4th Year\\4YP\\Pyro practice\\data\\{dims[1]}x{dims[2]}\\{dims[1]}x{dims[2]}_{num}_samples\\{dims[1]}x{dims[2]}_{num}_samples{i}.csv")
      data = read.csv(data_path)
      
      labels = colnames(data)
      
      df = as.data.frame(data)
      num_grids = length(labels)/2
      nlev = c(rep(4,num_grids),rep(5,num_grids))
      suffStat = list(dm=df, nlev=nlev, adaptDF=FALSE)
      for (p in sig_levels) {
        output3 = fci(suffStat=suffStat,indepTest = disCItest, alpha=p, labels=labels)
        
        adj_matrix = as(output3, "amat")
        adj_matrix_df = amat_to_dataframe(adj_matrix,labels,dims)

        ntests = sum(output3@n.edgetests) + sum(output3@n.edgetestsPDSEP)
        
        result = list(adj_matrix_df,'fci',ntests,'G2',p)
        
        # write_path = str_glue("C:\\Users\\order\\Documents\\Oxford\\4th Year\\4YP\\Pyro Practice\\data\\{dims[1]}x{dims[2]}\\R outputs\\outputs_{dims[1]}x{dims[2]}_{num}_samples\\{dims[1]}x{dims[2]}_{num}_samples{i}\\Constraint based\\{dims[1]}x{dims[2]}_{num}_samples{i}_fci_g2_{p}.csv")
        write.csv(result, str_glue("C:\\Users\\order\\Documents\\Oxford\\4th Year\\4YP\\Pyro Practice\\data\\{dims[1]}x{dims[2]}\\R outputs\\outputs_{dims[1]}x{dims[2]}_{num}_samples\\{dims[1]}x{dims[2]}_{num}_samples{i}\\Constraint based\\{dims[1]}x{dims[2]}_{num}_samples{i}_fci_g2_{p}.csv"))
      }
    }
  }
}

# plot(output3)
# adj_matrix