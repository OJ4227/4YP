# pcalg_algorithm GES

rm(list=ls())
setwd("C:/Users/order/Documents/Oxford/4th Year/4YP/R")
library(pcalg)
library(stringr)
library(Rgraphviz)

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
list_of_dims = list(c(3,3))
num_samples = c(100,200,500,1000,2000,5000,10000,20000,50000)
num_samples = c(500)
x = c(1:10)
x = c(1)


for (dims in list_of_dims) {
  for (num in num_samples) {
    for (i in x) {
      
      # Load in and format data
      data_path = str_glue("C:\\Users\\order\\Documents\\Oxford\\4th Year\\4YP\\Pyro practice\\data\\{dims[1]}x{dims[2]}\\{dims[1]}x{dims[2]}_{num}_samples\\{dims[1]}x{dims[2]}_{num}_samples{i}.csv")
      data = read.csv(data_path)
      
      labels = colnames(data)
      
      df = as.data.frame(data)
      
      score <- new("GaussL0penObsScore", df)
      output4 = pcalg::ges(score)
       
    }
  }
}

plot(output4$essgraph)
