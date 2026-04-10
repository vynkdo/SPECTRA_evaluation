library(readxl)
library(ggpubr)
library(tidyr)
library(reshape)

df <- data.frame(read_excel("./rsu_ensemble.xlsx", 
                                     sheet = "baselines_mr",
                                     col_types=c("text", "text", 
                                     "numeric", "text", "numeric")))
head(df)

regression_tasks = c("freesolv","lipo","delaney")

comparisons <- list(
  c("random", "scaffold"),
  c("random", "umap"),
  c("scaffold", "umap")
)

regression_tasks = c("freesolv","lipo","delaney")


for (model in c("LogReg", "RF", "XGB", "SVM")) { #classification models
  df_model = df[which(df$model == model),]

  p_classification <- ggboxplot(
    df_model,
    x = "split_type",
    y = "metric",
    fill = "split_type",
    # palette = "npg", # remove default palette
    width = 0.5,
    outlier.shape = NA
  ) +
    scale_fill_manual(
      values=c("orange","green4","red")
    ) +
    stat_compare_means(
      comparisons = comparisons,
      method = "wilcox.test",
      paired = FALSE,
      label = "p.signif",
      step.increase = 0.1
    ) +
    labs(
      x = "Split Type",
      y = "ROC AUC"
    ) +
    ggtitle(paste0("Performance: ", model)) + 
    theme_pubr(base_size = 14) +
    theme(
      legend.position = "right",
      axis.title = element_text(face = "bold")
    ) + facet_wrap(~dataset) 
  
  print(p_classification)
}



for (model in c("LinReg", "KRR")) { #regression models
  df_model = df[which(df$model == model),]
  tmp_df = df_model[which(df_model$dataset %in% regression_tasks),]
  p_regression <- ggboxplot(
    tmp_df,
    x = "split_type",
    y = "metric",
    fill = "split_type",
    # palette = "npg", # remove default palette
    width = 0.5,
    outlier.shape = NA
  ) +
    scale_fill_manual(
      values=c("orange","green4","red")
    ) +
    stat_compare_means(
      comparisons = comparisons,
      method = "wilcox.test",
      paired = FALSE,
      label = "p.signif",
      step.increase = 0.1
    ) +
    labs(
      x = "Split Type",
      y = "RMSE"
    ) +
    ggtitle(paste0("Performance: ", model)) + 
    theme_pubr(base_size = 14) +
    theme(
      legend.position = "right",
      axis.title = element_text(face = "bold")
    ) + facet_wrap(~dataset, scales="free_y") # bc RMSE on diff scale
  
  print(p_regression)
}
