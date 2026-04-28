library(dplyr)
library(ggplot2)
library(reticulate)


datasets <- c("bace", "bbbp","clintox","sider","tox21","lipo","delaney","freesolv")
base_path <- "/Users/ivymac/Desktop/SAGE_Lab/data_splitting_strategies"

pickle <- import("pickle")
read_pickle <- function(path) {
  py_run_string("
import pickle
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
")
  py$load_pickle(path) |> py_to_r()
}

load_dataset <- function(dataset_name) {
  SP_path <- file.path(base_path, "raw_splits", "spectra_tanimoto", paste0(dataset_name, "_SPECTRA_splits"))
  df_list <- list()
  for (p in 0:20) {
    SP <- sprintf("%.2f", p / 20)
    for (i in 0:2) {
      SP_dir <- file.path(SP_path, paste0("SP_",SP, "_", i))
      stats_file <- file.path(SP_dir, "stats.pkl")
      if (!file.exists(stats_file)) next
      stat_info <- read_pickle(stats_file)
      df_list[[length(df_list) + 1]] <- as.data.frame(stat_info)
    }
  }
  df <- bind_rows(df_list)
  
  avg_df <- df %>%
    group_by(SPECTRA_parameter) %>%
    summarise(cross_split_overlap_avg = mean(cross_split_overlap),
              cross_split_overlap_sd = sd(cross_split_overlap))
  
  output <- file.path(base_path, "splits_data","cross_split_overlap","spectra_tanimoto", paste0(dataset_name, "_spectra_cross_split_overlap.csv"))
  dir.create(dirname(output))
  write.csv(avg_df, output)
  avg_df$dataset <- dataset_name
  
  random_df <- read.csv(file.path(base_path, "splits_data/cross_split_overlap/random", paste0(dataset_name, "_random_cross_split_overlap.csv")))
  scaffold_df <- read.csv(file.path(base_path,"splits_data/cross_split_overlap/scaffold", paste0(dataset_name, "_scaffold_cross_split_overlap.csv")))
  umap_df <- read.csv(file.path(base_path, "splits_data/cross_split_overlap/umap", dataset_name, paste0(dataset_name, "_umap_cross_split_overlap.csv")))
  
  avg_df$random_mean <- mean(random_df$cross_split_overlap)
  avg_df$scaffold_mean <- mean(scaffold_df$cross_split_overlap)
  avg_df$umap_mean <- mean(umap_df$cross_split_overlap)
  
  return(avg_df)
}

all_data <- bind_rows(lapply(datasets, load_dataset))

p <- ggplot(all_data, aes(x = SPECTRA_parameter, y = cross_split_overlap_avg)) +
  
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
ggsave(filename = file.path(base_path, "plot", "facet_cross_split_overlap.png"), plot = p, dpi = 400, width = 13, height = 6)