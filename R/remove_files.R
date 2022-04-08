# Remove files
rm(list=ls())
list_of_dims = list(c(2,2),c(3,3),c(4,4),c(2,4),c(4,2))
num_samples = c(100,200,500,1000,2000,5000,10000,20000,50000)
x = c(1:10)

for (dims in list_of_dims) {
  for (num in num_samples) {
    for (i in x) {
      file_path = str_glue("C:\\Users\\order\\Documents\\Oxford\\4th Year\\4YP\\Pyro practice\\data\\{dims[1]}x{dims[2]}\\R outputs\\outputs_{dims[1]}x{dims[2]}_{num}_samples\\{dims[1]}x{dims[2]}_{num}_samples{i}\\rsmax2")
      do.call(file.remove, list(list.files(file_path, full.names = TRUE)))
    }
  }
}