# bnlearn algorithms3
rm(list=ls())

library(bnlearn)
library(stringr)

setwd("C:/Users/order/Documents/Oxford/4th Year/4YP/R")


list_of_dims = list(c(2,2),c(3,3),c(4,4),c(2,4),c(4,2))
num_samples = c(100,200,500,1000,2000,5000,10000,20000,50000)
x = c(1:10)


# Initialize vectors of tests and scores
indep_test =c('x2','mi')
scores = c('bic','bde')
restricts = c('mmpc','si.hiton.pc','hpc','iamb','fast.iamb','inter.iamb','iamb.fdr','gs','pc.stable')
# restricts = c('iamb','fast.iamb','inter.iamb','iamb.fdr','gs','pc.stable')
maximiser = c('hc','tabu')
sig_levels = c(0.01,0.05,0.1)

# dims=c(2,2)
# num = 100
for (dims in list_of_dims) {
  for (num in num_samples) {
    for (i in x) {
      
      # Load in and format data
      data_path = str_glue("C:\\Users\\order\\Documents\\Oxford\\4th Year\\4YP\\Pyro practice\\data\\{dims[1]}x{dims[2]}\\{dims[1]}x{dims[2]}_{num}_samples\\{dims[1]}x{dims[2]}_{num}_samples{i}.csv")
      data = read.csv(data_path)
      
      data = lapply(data, as.factor)
      
      df = as.data.frame(data)
      
      ## Constraint-based:
      # PC
      
      # alg_results_constraint = list()
      # 
      # for (test in indep_test) {
      #   for (p in sig_levels) {
      #     alg_results_constraint[[paste0("pc_",test,"_",p)]] = bnlearn::pc.stable(df,test = test,alpha = p)
      #     
      #     alg_results_constraint[[paste0("gs_",test,"_",p)]] = bnlearn::gs(df,test = test,alpha = p)
      #     
      #     alg_results_constraint[[paste0("iamb_",test,"_",p)]] = bnlearn::iamb(df,test = test,alpha = p)
      #     
      #     alg_results_constraint[[paste0("fast_iamb_",test,"_",p)]] = bnlearn::fast.iamb(df,test = test,alpha = p)
      #     
      #     alg_results_constraint[[paste0("inter_iamb_",test,"_",p)]] = bnlearn::inter.iamb(df,test = test,alpha = p)
      #     
      #     alg_results_constraint[[paste0("fdr_iamb_",test,"_",p)]] = bnlearn::iamb.fdr(df,test = test,alpha = p)
      #   }
      # }
      # 
      # ## Score based:
      # alg_results_score = list()
      # alg_results_hybrid = list()
      # for (score in scores) {
      #   alg_results_score[[paste0("hc_",score)]] = bnlearn::hc(df,score=score)
      #   alg_results_score[[paste0("tabu_",score)]] = bnlearn::tabu(df,score=score)
      #   
      #   ## Hybrid:
      #   for (test in indep_test) {
      #     for (p in sig_levels) {
      #       alg_results_hybrid[[paste0("mmhc_",test,"_",p,"_",score)]] = bnlearn::mmhc(df,maximize.args = score,restrict.args = list(test=test,alpha=p))
      #       alg_results_hybrid[[paste0("hp2c_",test,"_",p,"_",score)]] = bnlearn::h2pc(df,maximize.args = score,restrict.args = list(test=test,alpha=p))
      #     }
      #   }
      # }
    
      # rsmax2
      alg_results_rsmax2 = list()
      for (restrict in restricts) {
        for (maximize in maximiser) {
          for (score in scores) {
            for (test in indep_test) {
              for (p in sig_levels) {
                alg_results_rsmax2[[paste0("rsmax2_",restrict,'_',test,"_",p,"_",maximize,"_",score)]] = bnlearn::rsmax2(df,restrict = restrict,restrict.args = list(test=test,alpha=p),maximize = maximize,maximize.args = score)
              }
            }
          }
        }
      }
      
      # for (alg in names(alg_results_constraint)) { # Need to iterate through the names so we can name the files
      #   result = list(amat(alg_results_constraint[[alg]]),alg_results_constraint[[alg]]$learning$algo,alg_results_constraint[[alg]]$learning$ntests,alg_results_constraint[[alg]]$learning$test,alg_results_constraint[[alg]]$learning$args$alpha)
      #   write.csv(result,str_glue("C:\\Users\\order\\Documents\\Oxford\\4th Year\\4YP\\Pyro Practice\\data\\{dims[1]}x{dims[2]}\\R outputs\\outputs_{dims[1]}x{dims[2]}_{num}_samples\\{dims[1]}x{dims[2]}_{num}_samples{i}\\Constraint based\\{dims[1]}x{dims[2]}_100_samples{i}_{alg}.csv"))
      # } 
      # 
      # for (alg in names(alg_results_score)) { # Need to iterate through the names so we can name the files
      #   result = list(amat(alg_results_score[[alg]]),alg_results_score[[alg]]$learning$algo,alg_results_score[[alg]]$learning$ntests,alg_results_score[[alg]]$learning$test)
      #   write.csv(result,str_glue("C:\\Users\\order\\Documents\\Oxford\\4th Year\\4YP\\Pyro Practice\\data\\{dims[1]}x{dims[2]}\\R outputs\\outputs_{dims[1]}x{dims[2]}_{num}_samples\\{dims[1]}x{dims[2]}_{num}_samples{i}\\Score based\\{dims[1]}x{dims[2]}_100_samples{i}_{alg}.csv"))
      # } 
      # 
      # for (alg in names(alg_results_hybrid)) { # Need to iterate through the names so we can name the files
      #   result = list(amat(alg_results_hybrid[[alg]]),alg_results_hybrid[[alg]]$learning$algo,alg_results_hybrid[[alg]]$learning$ntests,alg_results_hybrid[[alg]]$learning$rstest,alg_results_hybrid[[alg]]$learning$args$alpha,alg_results_hybrid[[alg]]$learning$test)
      #   write.csv(result,str_glue("C:\\Users\\order\\Documents\\Oxford\\4th Year\\4YP\\Pyro Practice\\data\\{dims[1]}x{dims[2]}\\R outputs\\outputs_{dims[1]}x{dims[2]}_{num}_samples\\{dims[1]}x{dims[2]}_{num}_samples{i}\\Hybrid\\{dims[1]}x{dims[2]}_100_samples{i}_{alg}.csv"))
      # } 
      
      for (alg in names(alg_results_rsmax2)) { # Need to iterate through the names so we can name the files
        result = list(amat(alg_results_rsmax2[[alg]]),alg_results_rsmax2[[alg]]$learning$algo,alg_results_rsmax2[[alg]]$learning$ntests,alg_results_rsmax2[[alg]]$learning$restrict,alg_results_rsmax2[[alg]]$learning$rstest,alg_results_rsmax2[[alg]]$learning$args$alpha,alg_results_rsmax2[[alg]]$learning$maximize,alg_results_rsmax2[[alg]]$learning$maxscore)
        write.csv(result,str_glue("C:\\Users\\order\\Documents\\Oxford\\4th Year\\4YP\\Pyro Practice\\data\\{dims[1]}x{dims[2]}\\R outputs\\outputs_{dims[1]}x{dims[2]}_{num}_samples\\{dims[1]}x{dims[2]}_{num}_samples{i}\\rsmax2\\{dims[1]}x{dims[2]}_100_samples{i}_{alg}.csv"))
      } 
      
    }
  }
}

