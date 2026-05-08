library(dplyr)
library(ggplot2)
library(reticulate)


datasets <- c("bace", "bbbp","clintox","sider","tox21","lipo","delaney","freesolv")
base_path <- "/Users/ivymac/Desktop/SAGE_Lab/data_splitting_strategies"

load_dataset <- function(dataset_name) {
  SP_df <- read.csv(file.path(base_path, "splits_data", "cross_split_overlap", "spectra_tanimoto", paste0(dataset_name, "_spectra_tanimoto_cross_split_overlap.csv")))
  avg_df <- SP_df %>%
    group_by(SPECTRA_parameter) %>%
    summarise(spectra_mean = mean(cross_split_overlap))
  
  random_df <- read.csv(file.path(base_path, "splits_data/cross_split_overlap/random", paste0(dataset_name, "_random_cross_split_overlap.csv")))
  scaffold_df <- read.csv(file.path(base_path,"splits_data/cross_split_overlap/scaffold", paste0(dataset_name, "_scaffold_cross_split_overlap.csv")))
  umap_df <- read.csv(file.path(base_path, "splits_data/cross_split_overlap/umap", paste0(dataset_name, "_umap_cross_split_overlap.csv")))
  
  avg_df$random_mean <- mean(random_df$cross_split_overlap)
  avg_df$scaffold_mean <- mean(scaffold_df$cross_split_overlap)
  avg_df$umap_mean <- mean(umap_df$cross_split_overlap)
  
  avg_df$dataset <- dataset_name
  return(avg_df)
}

all_data <- bind_rows(lapply(datasets, load_dataset))

p <- ggplot(all_data, aes(x = SPECTRA_parameter, y = spectra_mean)) +
  
  geom_point(aes(color = "SPECTRA"), size = 2) +
  geom_hline(aes(yintercept = random_mean, color = "Random"), linetype = "solid", size = 0.6) +
  geom_hline(aes(yintercept = scaffold_mean, color = "Scaffold"), linetype = "dashed", size = 0.6) +
  geom_hline(aes(yintercept = umap_mean, color = "UMAP"), linetype = "dotdash", size = 0.6) +
  
  facet_wrap(~dataset, nrow = 2) +
  
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.1)) +
  scale_y_continuous(limits = c(0, 0.20), breaks = seq(0, 0.2, 0.02)) +
  
  scale_color_manual(
    name = "Split Method",
    breaks = c("SPECTRA", "Random", "Scaffold", "UMAP"),
    values = c(
      SPECTRA = "gray50",
      Random = "green4",
      Scaffold = "orange",
      UMAP = "red")) +
  
  labs(
    x = "Spectral parameter",
    y = "Cross-split overlap") +
  
  theme_classic() +
  theme(
    strip.text = element_text(face = "bold"),
    legend.position = "right",
    legend.title = element_text(size = 10, face = "bold"),
    legend.text = element_text(size = 10),
    legend.background = element_rect(fill = "white", color = "black", linewidth = 0.6),
    legend.key = element_rect(fill = "white"),
    axis.title.x = element_text(size = 12),
    axis.title.y = element_text(size = 12)
  )

print(p)
ggsave(filename = file.path(base_path, "plot", "facet_cross_split_overlap_new.png"), plot = p, dpi = 400, width = 13, height = 6)