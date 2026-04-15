library(ggpubr)
library(ggbeeswarm)
library(tidyr)

my_colors = c("green4", "orange","red")

create_plot <- function(data, model_name, metric_name, my_colors, scales="fixed") {
  p <- ggplot(
    data,
    aes(split_type, metric, color=split_type),
  ) +
    geom_quasirandom() +
    scale_color_manual(
      values=my_colors
    ) +
    stat_compare_means(
      comparisons = comparisons,
      method = "wilcox.test",
      paired = FALSE, 
      label = "p.signif",
      step.increase = 0.1,
      vjust=0.5,
      hide.ns=TRUE
    ) +
    labs(
      x = "Split Type",
      y = metric_name
    ) +
    theme_pubr(base_size = 14) +
    theme(
      legend.position = "none",
      axis.title = element_text(face = "bold")
    ) + facet_wrap(~dataset, scales=scales) +
    ggtitle(model_name)
  
  print(p)
}
