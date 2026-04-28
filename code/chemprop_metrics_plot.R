library(dplyr)
library(ggplot2)

datasets <- c("bace","bbbp","clintox","sider","tox21")
base_path <- "/Users/ivymac/Desktop/SAGE_Lab/data_splitting_strategies"

load_dataset <- function(dataset_name) {
  rsu <- read.csv(file.path(base_path, "metrics", "rsu_classification", paste0(dataset_name, ".csv")))
  spectra_cso <- read.csv(file.path(base_path, "splits_data", "cross_split_overlap", "spectra_tanimoto", paste0(dataset_name, "_spectra_cross_split_overlap.csv")))
  spectra_metrics <- read.csv(file.path(base_path, "metrics", "spectra_tanimoto", "avg_metrics", paste0(dataset_name, ".csv")))
  
  spectra_df <- merge(spectra_metrics, spectra_cso, by = "SPECTRA_parameter")
  spectra_df$split <- 'SPECTRA'
  df <- bind_rows(spectra_df, rsu)
  df$dataset <- dataset_name
  
  return(df)
}

all_data <- bind_rows(lapply(datasets, load_dataset))

p <- ggplot(all_data, aes(x = cross_split_overlap_avg, y = AUC)) +
  
  geom_point(aes(color = split, shape = split), size = 3) +
  geom_errorbar(aes(ymin = AUC - SD, ymax = AUC + SD, color = split), width = 0.001, size = 0.4) +
  geom_errorbarh(aes(xmin = cross_split_overlap_avg - cross_split_overlap_sd, xmax = cross_split_overlap_avg + cross_split_overlap_sd, color = split), height = 0.01, linetype = "dashed", size = 0.4)+
  facet_wrap(~dataset, scale = "free_x", nrow = 2) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.1)) +
  
  scale_shape_manual(
    name = "Split Method",
    breaks = c("SPECTRA", "Random", "Scaffold", "UMAP"),
    values = c(
      SPECTRA = 16,
      Random = 15,
      Scaffold = 17,
      UMAP = 18)) +
  
  scale_color_manual(
    name = "Split Method",
    breaks = c("SPECTRA", "Random", "Scaffold", "UMAP"),
    values = c(
      SPECTRA = "gray50",
      Random = "green4",
      Scaffold = "orange",
      UMAP = "red")) +
  
  labs(
    x = "Cross-split overlap",
    y = "ROC-AUC") +
  
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
ggsave(filename = file.path(base_path, "plot", "chemprop_classification.png"), plot = p, dpi = 400, width = 13, height = 8)