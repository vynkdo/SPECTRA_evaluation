library(readxl)
library(tidyr)
library(reshape)
source("./plot_helper_func.R")

classical_df <- data.frame(read_excel("./rsu_ensemble.xlsx", 
                                     sheet = "classical_baselines",
                                     col_types=c("text", "text", 
                                     "numeric", "text", "numeric")))
head(classical_df)
classical_df$ensemble <- NA

chemprop_df <- data.frame(read_excel("./rsu_ensemble.xlsx", 
                                     sheet = "random_scaffold_umap",
                                     col_types=c("text", "numeric", "numeric", "numeric",
                                                 "numeric", "numeric")))
head(chemprop_df)

chemprop_df <- data.frame(chemprop_df %>% pivot_longer( 
  cols=c("random", "scaffold", "umap"), 
  names_to="split_type", values_to="metric"))

head(chemprop_df)
chemprop_df$model = "chemprop"

df <- rbind(chemprop_df, classical_df)
head(df)


regression_tasks = c("freesolv","lipo","delaney")

comparisons <- list(
  c("random", "scaffold"),
  c("random", "umap"),
  c("scaffold", "umap")
)


for (model in c("LogReg", "RF", "XGB", "SVM", "chemprop")) { #classification models
  df_model = df[which(df$model == model),]
  if (length(which(df_model$dataset %in% regression_tasks)) > 0) {
    df_model = df_model[-which(df_model$dataset %in% regression_tasks),]
  }
  create_plot(df_model, model,"AUC", my_colors)
}



for (model in c("LinReg", "KRR", "chemprop")) { #regression models
  df_model = df[which(df$model == model),]
  df_model = df_model[which(df_model$dataset %in% regression_tasks),]
  create_plot(df_model, model, "RMSE",my_colors, scales="free_y")
}
