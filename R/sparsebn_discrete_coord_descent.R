# discrete coordinate descent algorithm

library(discretecdAlgorithm)
library(sparsebn)
library(sparsebnUtils)
library(stringr)

dims = c(2,2)

num = 500
x = c(1:10)
x = c(1)

data_path = str_glue("C:\\Users\\order\\Documents\\Oxford\\4th Year\\4YP\\Pyro practice\\data\\{dims[1]}x{dims[2]}\\{dims[1]}x{dims[2]}_{num}_samples\\{dims[1]}x{dims[2]}_{num}_samples{i}.csv")
data = read.csv(data_path)

labels = colnames(data)

df = as.data.frame(data)
bn_data = sparsebnData(df, type = 'discrete')

start_time = Sys.time()
output1 = discretecdAlgorithm::cd.run(bn_data)

idx = select.parameter(output1, bn_data, alpha = 0.1)
get.adjacency.matrix(output1[[idx]])
end_time = Sys.time()
plot(output1[[idx]])
duration = end_time - start_time
print(duration)