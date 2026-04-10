library(readxl)
library(ggpubr)
library(tidyr)
library(reshape)

splits_data <- data.frame(read_excel("./rsu_ensemble.xlsx", 
                           sheet = "machine_readable",
                           col_types=c("text", "numeric", "numeric", "numeric",
                                       "numeric", "numeric")))
head(splits_data)

df <- splits_data %>% pivot_longer( 
  cols=c("random", "scaffold", "umap"), 
  names_to="split_type", values_to="value")

head(df)

regression_tasks = c("freesolv","lipo","delaney")

comparisons <- list(
  c("random", "scaffold"),
  c("random", "umap"),
  c("scaffold", "umap")
)

p_classification <- ggboxplot(
  df[-which(df$dataset %in% regression_tasks),],
  x = "split_type",
  y = "value",
  fill = "split_type",
  #palette = "npg", # decent default palette
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
    y = "AUC"
  ) +
  theme_pubr(base_size = 14) +
  theme(
    legend.position = "right",
    axis.title = element_text(face = "bold")
  ) + facet_wrap(~dataset)

p_classification



p_regression <- ggboxplot(
  df[which(df$dataset %in% regression_tasks),],
  x = "split_type",
  y = "value",
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
  theme_pubr(base_size = 14) +
  theme(
    legend.position = "right",
    axis.title = element_text(face = "bold")
  ) + facet_wrap(~dataset, scales="free_y") # bc RMSE on diff scale

p_regression
