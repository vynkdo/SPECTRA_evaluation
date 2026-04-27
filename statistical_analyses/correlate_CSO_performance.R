library(readxl)
library(tidyr)
library(reshape)

classical_df <- data.frame(read_excel("./statistical_analyses/rsu_ensemble.xlsx", 
                                      sheet = "classical_baselines",
                                      col_types=c("text", "text", 
                                                  "numeric", "text", "numeric")))
head(classical_df)


cso <- read.csv(paste0("./splits_data/cross_split_overlap/umap/clintox/",
                "clintox_umap_cross_split_overlap.csv"))

for (model in c("SVM", "RF", "XGB", "LogReg")) {
  perf <- classical_df[which(classical_df$dataset == "clintox" & classical_df$split_type == "umap"
                     & classical_df$model == model),"metric"]
  
  plot(perf ~ cso$cross_split_overlap, main=model)
  print(model)
  print(cor.test(perf, cso$cross_split_overlap))
}
