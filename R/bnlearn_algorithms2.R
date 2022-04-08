# bnlearn algorithms2
rm(list=ls())

library(bnlearn)
library(stringr)

setwd("C:/Users/order/Documents/Oxford/4th Year/4YP/R")


list_of_dims = list(c(2,2),c(3,3),c(4,4),c(2,4),c(4,2))
num_samples = c(100,200,500,1000,2000,5000,10000,20000,50000)
x = c(1:10)
alg_results = list()

indep_test =c('x2','mi')
scores = c('bic','bde')
restricts = c('mmpc','si.hiton.pc','hpc')
maximiser = c('hc','tabu')
sig_levels = c(0.01,0.05,0.1)
dims=c(2,2)
num = 100
# for (dims in list_of_dims) {
  # for (num in num_samples) {
    for (i in x) {
      
      # Load in and format data
      data_path = str_glue("C:\\Users\\order\\Documents\\Oxford\\4th Year\\4YP\\Pyro practice\\data\\{dims[1]}x{dims[2]}\\{dims[1]}x{dims[2]}_{num}_samples\\{dims[1]}x{dims[2]}_{num}_samples{x[i]}.csv")
      data = read.csv(data_path)
    
      data = lapply(data, as.factor)
      
      df = as.data.frame(data)
      
      ## Constraint-based:
      # PC
      
      for (test in indep_test) {
        alg_results[[paste0("pc_",test)]] = bnlearn::pc.stable(df,test = test)
        
        alg_results[[paste0("gs_",test)]] = bnlearn::gs(df,test = test)
        
        alg_results[[paste0("iamb_",test)]] = bnlearn::iamb(df,test = test)
        
        alg_results[[paste0("fast_iamb_",test)]] = bnlearn::fast.iamb(df,test = test)
        
        alg_results[[paste0("inter_iamb_",test)]] = bnlearn::inter.iamb(df,test = test)
        
        alg_results[[paste0("fdr_iamb_",test)]] = bnlearn::iamb.fdr(df,test = test)
        
      }
      
      ## Score based:
      
      for (score in scores) {
        alg_results[[paste0("hc_",score)]] = bnlearn::hc(df,score=score)
        alg_results[[paste0("tabu_",score)]] = bnlearn::tabu(df,score=score)
        
      ## Hybrid:
        for (test in indep_test) {
          alg_results[[paste0("mmhc_",test,"_",score)]] = bnlearn::mmhc(df,maximize.args = score,restrict.args = list(test=test))
          alg_results[[paste0("hp2c_",test,"_",score)]] = bnlearn::h2pc(df,maximize.args = score,restrict.args = list(test=test))
        }
      }
      
      ## Hybrid:
      # 
      # for (score in scores) {
      #   alg_results[[paste0("mmhc_",score)]] = bnlearn::mmhc(df,maximize.args = score)
      #   alg_results[[paste0("hp2c_",score)]] = bnlearn::h2pc(df,maximize.args = score)
      # }
      
      # resmax2
      
      for (restrict in restricts) {
        for (maximize in maximiser) {
          for (score in scores) {
            alg_results[[paste0("resmax2_",restrict,"_",maximize,"_",score)]] = bnlearn::rsmax2(df,restrict = restrict,maximize = maximize,maximize.args = score)
          }
        }
      }
      
      for (alg in names(alg_results)) {
        result = list(amat(alg_results[[alg]]),alg_results[[alg]]$learning$ntests)
        # write.csv(result,str_glue("C:\\Users\\order\\Documents\\Oxford\\4th Year\\4YP\\Pyro Practice\\data\\{dims[1]}x{dims[2]}\\R outputs\\{alg$learning$}_{dims[1]}x{dims[2]}_100_samples{x[i]}.csv"))
      }
      # write.csv(adj_gs, str_glue("C:\\Users\\order\\Documents\\Oxford\\4th Year\\4YP\\Pyro Practice\\adj_gs_2x2_100_samples{x[i]}.csv"))
      # write.csv(adj_hc, str_glue("C:\\Users\\order\\Documents\\Oxford\\4th Year\\4YP\\Pyro Practice\\adj_hc_2x2_100_samples{x[i]}.csv"))
      # write.csv(adj_mmhc, str_glue("C:\\Users\\order\\Documents\\Oxford\\4th Year\\4YP\\Pyro Practice\\adj_mmhc_2x2_100_samples{x[i]}.csv"))
      # 
      
    }
  # }
# }


# rsmax2_alg = bnlearn::rsmax2(df)
# adj_rsmax2 = bnlearn::amat(rsmax2_alg)

# data = read.csv("2x2_100_samples1.csv")
# labels = colnames(data)
# 
# data = lapply(data, as.factor)
# 
# df = as.data.frame(data)
# output = bnlearn::gs(df,test = 'x2')
# output = bnlearn::hc(df)
# output = bnlearn::mmhc(df)
# output = bnlearn::hc(df)

# library(Rgraphviz)
# graphviz.plot(mmhc_alg, layout = "dot")
