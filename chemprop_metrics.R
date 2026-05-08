library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)

datasets <- c("delaney", "freesolv", "lipo")
base_path <- "/Users/ivymac/Desktop/SAGE_Lab/data_splitting_strategies"

spectra_process <- function(dataset_name) {
  cso_df <- read.csv(file.path(base_path,"splits_data","cross_split_overlap","spectra_tanimoto", paste0(dataset_name,"_spectra_tanimoto_cross_split_overlap.csv")))
  metrics_df <- read.csv(file.path(base_path,"metrics","spectra_tanimoto", "sheet", paste0("spectra_ensemble_",dataset_name,".csv")))
  
  metrics_df <- metrics_df %>%
    mutate(SPECTRA_parameter = as.numeric(str_extract(RMSE, "^[0-9.]+")))
  
  metrics_vals <- metrics_df %>%
    pivot_longer(cols = starts_with("Ensemble"),
                 values_to = "metrics") %>%
    group_by(SPECTRA_parameter) %>%
    filter(sum(!is.na(metrics)) == 15) %>%
    summarise(metrics_mean = mean(metrics, na.rm = TRUE),
              metrics_sd = sd(metrics, na.rm = TRUE))
  
  cso_vals <- cso_df %>%
    group_by(SPECTRA_parameter) %>%
    summarise(cso_mean = mean(cross_split_overlap),
              cso_sd = sd(cross_split_overlap))
  
  spectra_summary <- left_join(cso_vals, metrics_vals, by = "SPECTRA_parameter")
  spectra_summary$split <- "SPECTRA"
  spectra_summary$dataset <- dataset_name
  
  return(spectra_summary)
}

rsu_process <- function(dataset_name, split_type) {
  cso_df <- read.csv(file.path(base_path,"splits_data","cross_split_overlap",split_type, paste0(dataset_name,"_", split_type, "_cross_split_overlap.csv")))
  metrics_df <- read.csv(file.path(base_path,"metrics", split_type, paste0(split_type, "_metrics_", dataset_name, ".csv")))
  
  cso_stats <- cso_df %>%
    summarise(
      cso_mean = mean(cross_split_overlap),
      cso_sd = sd(cross_split_overlap)
    )
  
  metrics_list <- unlist(metrics_df[, grep("Ensemble", names(metrics_df))])
  metrics_stats <- data.frame(
    metrics_mean = mean(metrics_list),
    metrics_sd = sd(metrics_list))
  
  rsu_df <- bind_cols(cso_stats, metrics_stats)
  rsu_df$split <- split_type
  rsu_df$dataset <- dataset_name
  
  return(rsu_df)
}

map_to_spectra <- function(rsu_df, spectra_df) {
  spectra_df <- spectra_df %>%
    filter(!is.na(metrics_mean))
  rsu_df <- rsu_df %>%
    rowwise() %>%
    mutate(SPECTRA_parameter = spectra_df$SPECTRA_parameter[which.min(abs(cso_mean - spectra_df$cso_mean))])
  
  final_df <- bind_rows(rsu_df, spectra_df)
  return(final_df)
}
  
all_data <- bind_rows(lapply(datasets, function(d){
  spectra_df <- spectra_process(d)
  rsu_list <- lapply(c("random", "scaffold", "umap"), function(s) rsu_process(d, s))
  rsu_df <- bind_rows(rsu_list)
  final_df <- map_to_spectra(rsu_df, spectra_df)
  return(final_df)
}))
all_data <- all_data %>% mutate(split = recode(split, "random" = "Random", "scaffold" = "Scaffold", "umap" = "UMAP"))
write.csv(all_data, file = file.path(base_path, "metrics","regression_summary.csv"))
p <- ggplot(all_data, aes(x = cso_mean, y = metrics_mean)) +
  geom_point(aes(color = split, shape = split), size = 3) +
  geom_errorbar(aes(ymin = metrics_mean - metrics_sd, ymax = metrics_mean + metrics_sd, color = split), width = 0.001, size = 0.4) +
  geom_errorbarh(aes(xmin = cso_mean - cso_sd, xmax = cso_mean + cso_sd, color = split), height = 0.01, linetype = "dashed", size = 0.4) +
  facet_wrap(~dataset, nrow = 1, scales = "free") +
  #scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.1)) +
  scale_shape_manual(
    name = "Split Method",
    breaks = c("SPECTRA", "Random", "Scaffold", "UMAP"),
    values = c(
      SPECTRA = 16,
      Random = 15,
      Scaffold = 17,
      UMAP = 18
    )
  ) +
  scale_color_manual(
    name = "Split Method",
    breaks = c("SPECTRA", "Random", "Scaffold", "UMAP"),
    values = c(
      SPECTRA = "gray50",
      Random = "green4",
      Scaffold = "orange",
      UMAP = "red"
    )
  ) +
  labs(
    x = "Cross-split overlap",
    y = "RMSE"
  ) +
  
  theme_classic() +
  theme(
    strip.text = element_text(face = "bold"),
    panel.spacing = unit(1.5, "lines"),
    legend.position = "right",
    legend.title = element_text(size = 10, face = "bold"),
    legend.text = element_text(size = 10),
    legend.background = element_rect(fill = "white", color = "black", linewidth = 0.6),
    legend.key = element_rect(fill = "white"),
    axis.title.x = element_text(size = 12),
    axis.title.y = element_text(size = 12)
  )

print(p)
ggsave(filename = file.path(base_path, "plot", "chemprop_regression_new_1.png"), plot = p, dpi = 400, width = 13, height = 4)