library(readxl)
library(tidyr)
library(reshape)
source("./plot_helper_func.R")

splits_data <- data.frame(read_excel("./rsu_ensemble.xlsx", 
                           sheet = "random_scaffold_umap",
                           col_types=c("text", "numeric", "numeric", "numeric",
                                       "numeric", "numeric")))
head(splits_data)

df <- splits_data %>% pivot_longer( 
  cols=c("random", "scaffold", "umap"), 
  names_to="split_type", values_to="metric")

head(df)

regression_tasks = c("freesolv","lipo","delaney")

comparisons <- list(
  c("random", "scaffold"),
  c("random", "umap"),
  c("scaffold", "umap")
)

create_plot(df[-which(df$dataset %in% regression_tasks),], "AUC", my_colors)
create_plot(df[which(df$dataset %in% regression_tasks),], "RMSE", 
            my_colors, scales="free_y")
