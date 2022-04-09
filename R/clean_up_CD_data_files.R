# Clean up CD data files
rm(list=ls())
library(stringr)

dims = c(2,2)
num_samples = c(20000, 50000)
x = (1:10)
alphas = c(0.01, 0.05, 0.1)

for (num in num_samples) {
  for (i in x) {
    for (alpha in alphas) {
      file_path = str_glue("C:\\Users\\order\\Documents\\Oxford\\4th Year\\4YP\\Pyro practice\\data\\{dims[1]}x{dims[2]}\\R outputs\\outputs_{dims[1]}x{dims[2]}_{num}_samples\\{dims[1]}x{dims[2]}_{num}_samples{i}\\CD\\")
      file = str_glue("{dims[1]}x{dims[2]}_{num}_samples{i}_CD_{alpha}.csv")
      data = read.csv(file_path + file)
      
      data = data[-c(1)]
      data[10] = sapply(data[10], function(a) {a*60})
      
      write.csv(data, str_glue("C:\\Users\\order\\Documents\\Oxford\\4th Year\\4YP\\Pyro practice\\data\\CD_{alpha}.csv"))
    }
  }
}
