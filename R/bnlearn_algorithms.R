# bnlearn algorithms

library(bnlearn)
library(stringr)

setwd("C:/Users/order/Documents/Oxford/4th Year/4YP/R")


list_of_dims = list(c(2,2),c(3,3),c(4,4),c(2,4),c(4,2))
x = c(1,2)


for (i in x) {
  data_path = str_glue("C:\Users\order\Documents\Oxford\4th Year\4YP\Pyro practice\data\2x2\2x2_100_samples\2x2_100_samples{x[i]}.csv")
  data = read.csv(data_path)
  print(typeof(data))
  data = lapply(data, as.factor)
  
  df = as.data.frame(data)
  
  ## Constraint-based:
  # PC
  
  indep_test =c('x2','mi')
  sig_levels = c(0.01,0.05,0.1)
  for (test in indep_test) {
    assign(paste0("pc_",test),bnlearn::pc.stable(df,test = test))
    assign(paste0("gs_",test),bnlearn::gs(df,test = test))
    assign(paste0("iamb_",test),bnlearn::iamb(df,test = test))
    assign(paste0("fast_iamb_",test),bnlearn::fast.iamb(df,test = test))
    assign(paste0("inter_iamb_",test),bnlearn::inter.iamb(df,test = test))
    assign(paste0("fdr_iamb_",test),bnlearn::iamb.fdr(df,test = test))
    
  }
  pc_x2 = bnlearn::pc.stable(df, test = 'x2')
  adj_pc_x2 = bnlearn::amat(pc_x2)

  pc_mi = bnlearn::pc.stable(df, test =  'mi') # Might not need this as it is just proportional to G-square test
  adj_pc_mi = bnlearn::amat(pc_mi)
  
  # GS
  gs_x2 = bnlearn::gs(df,test = 'x2')
  adj_gs_x2 = bnlearn::amat(gs_x2)
  
  gs_mi = bnlearn::gs(df, test = 'mi')
  adj_gs_mi = bnlearn::amat(pc_mi)
  
  # IAMB and variants
  iamb_x2 = bnlearn::iamb(df, test = 'x2')
  adj_iamb_mi = bnlearn::amat(iamb_x2)
  
  iamb_mi = bnlearn::iamb(df, test = 'mi')
  adj_iamb_mi = bnlearn::amat(iamb_mi)
  
  fast_iamb_x2 = bnlearn::fast.iamb(df, test = 'x2')
  adj_fast_iamb_x2 = bnlearn::amat(fast_iamb_x2)
  
  fast_iamb_mi = bnlearn::fast.iamb(df, test = 'mi')
  adj_fast_iamb_mi = bnlearn::amat(fast_iamb_mi)

  inter_iamb_x2 = bnlearn::inter.iamb(df, test = 'x2')
  adj_inter_iamb_x2 = bnlearn::amat(inter_iamb_x2)
  
  inter_iamb_mi = bnlearn::inter.iamb(df, test = 'mi')
  adj_inter_iamb_mi = bnlearn::amat(inter_iamb_mi)
  
  fdr_iamb_x2 = bnlearn::iamb.fdr(df, test = 'x2')
  adj_fdr_iamb_x2 = bnlearn::amat(fdr_iamb_x2)
  
  fdr_iamb_mi = bnlearn::iamb.fdr(df, test = 'mi')
  adj_fdr_iamb_mi = bnlearn::amat(fdr_iamb_mi)
  
  ## Score based:
  scores = c('bic','bde')
  for (score in scores) {
    assign(paste0("hc_",score),bnlearn::hc(df,score=score))
    assign(paste0("tabu_",score),bnlearn::tabu(df,score=score))
  }
  
  hc_bic = bnlearn::hc(df, score = 'bic')
  adj_hc_bic = bnlearn::amat(hc_bic)
  
  hc_bdeu = bnlearn::hc(df, score = 'bde')
  adj_hc_bdeu = bnlearn::amat(hc_bdeu)
  
  tabu_bic = bnlearn::tabu(df, score = 'bic')
  adj_tabu_bic = bnlearn::amat(tabu_bic)
  
  tabu_bde = bnlearn::tabu(df, score = 'bde')
  adj_tabu_bde = bnlearn::amat(tabu_bde)
  
  ## Hybrid:
  
  scores = c('bic','bde')
  
  for (score in scores) {
    assign(paste0("mmpc_",score),bnlearn::mmhc(df,maximize.args = (score)))
    assign(paste0("hp2c_",score),bnlearn::h2pc(df,maximize.args = (score)))
  }
  # mmhc_alg = bnlearn::mmhc(df)
  adj_mmhc = bnlearn::amat(mmhc_alg)
  
  # hp2c_alg = bnlearn::h2pc(df)
  adj_h2pc = bnlearn::amat(h2pc_alg)
  
  
  restricts = c('mmpc','si.hiton.pc','hpc')
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
