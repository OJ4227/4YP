# bnlearn algorithms1
rm(list=ls())

library(bnlearn)
library(stringr)

setwd("C:/Users/order/Documents/Oxford/4th Year/4YP/R")


list_of_dims = list(c(2,2),c(3,3),c(4,4),c(2,4),c(4,2))
x = c(1)
alg_results = list()

for (i in x) {
  data_path = str_glue("C:\\Users\\order\\Documents\\Oxford\\4th Year\\4YP\\Pyro practice\\data\\2x2\\2x2_100_samples\\2x2_100_samples{x[i]}.csv")
  data = read.csv(data_path)
  print(typeof(data))
  data = lapply(data, as.factor)
  
  df = as.data.frame(data)
  
  ## Constraint-based:
  # PC
  
  indep_test =c('x2','mi')
  for (test in indep_test) {
    # alg_results = append(alg_results,assign(paste0("pc_",test),bnlearn::pc.stable(df,test = test))) # Might not need the pc_mi test as it is just proportional to G-square test
    # assign(paste0("pc_",test),bnlearn::pc.stable(df,test = test))
    # alg_results = append(alg_results,paste0("pc_",test)=list(assign(paste0("pc_",test),bnlearn::pc.stable(df,test = test))))
    alg_results[[paste0("pc_",test)]] = bnlearn::pc.stable(df,test = test)
    
    # alg_results = append(alg_results,assign(paste0("gs_",test),bnlearn::gs(df,test = test)))
    # assign(paste0("gs_",test),bnlearn::gs(df,test = test))
    # alg_results = append(alg_results,list(assign(paste0("gs_",test),bnlearn::gs(df,test = test))))
    alg_results[[paste0("gs_",test)]] = bnlearn::gs(df,test = test)
    
    # alg_results = append(alg_results,assign(paste0("iamb_",test),bnlearn::iamb(df,test = test)))
    # assign(paste0("iamb_",test),bnlearn::iamb(df,test = test))
    alg_results[[paste0("iamb_",test)]] = bnlearn::iamb(df,test = test)
    
    # alg_results = append(alg_results,assign(paste0("fast_iamb_",test),bnlearn::fast.iamb(df,test = test)))
    alg_results[[paste0("fast_iamb_",test)]] = bnlearn::fast.iamb(df,test = test)
    
    # alg_results = append(alg_results,assign(paste0("inter_iamb_",test),bnlearn::inter.iamb(df,test = test)))
    alg_results[[paste0("inter_iamb_",test)]] = bnlearn::inter.iamb(df,test = test)
    
    # alg_results = append(alg_results,assign(paste0("fdr_iamb_",test),bnlearn::iamb.fdr(df,test = test)))
    alg_results[[paste0("fdr_iamb_",test)]] = bnlearn::iamb.fdr(df,test = test)
    
  }
  
  ## Score based:
  scores = c('bic','bde')
  for (score in scores) {
    assign(paste0("hc_",score),bnlearn::hc(df,score=score))
    assign(paste0("tabu_",score),bnlearn::tabu(df,score=score))
  }
  
  ## Hybrid:
  
  scores = c('bic','bde')
  
  for (score in scores) {
    assign(paste0("mmpc_",score),bnlearn::mmhc(df,maximize.args = (score)))
    assign(paste0("hp2c_",score),bnlearn::h2pc(df,maximize.args = (score)))
  }

  # resmax2
  restricts = c('mmpc','si.hiton.pc','hpc','iamb','fast.iamb','inter.iamb','iamb.fdr','gs','pc.stable')
  # restricts = c('pc.stable','gs')
  maximiser = c('hc','tabu')
  
  for (restrict in restricts) {
    for (maximize in maximiser) {
      for (score in scores) {
        assign(paste0("resmax2_",restrict,"_",maximize,"_",score),bnlearn::rsmax2(df,restrict = restrict,maximize = maximize,maximize.args = list(score)))
      }
    }
  }
  
  
  # write.csv(adj_gs, str_glue("C:\\Users\\order\\Documents\\Oxford\\4th Year\\4YP\\Pyro Practice\\adj_gs_2x2_100_samples{x[i]}.csv"))
  # write.csv(adj_hc, str_glue("C:\\Users\\order\\Documents\\Oxford\\4th Year\\4YP\\Pyro Practice\\adj_hc_2x2_100_samples{x[i]}.csv"))
  # write.csv(adj_mmhc, str_glue("C:\\Users\\order\\Documents\\Oxford\\4th Year\\4YP\\Pyro Practice\\adj_mmhc_2x2_100_samples{x[i]}.csv"))
  # 
  
}

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
