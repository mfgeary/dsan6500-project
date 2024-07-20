library(tidyverse)
# Load the data
train_labels <- read_csv("data/train_label_coordinates.csv")

train_labels |>
    ggplot(aes(x = condition)) +
    geom_bar(fill = "blue4") +
    coord_flip() +
    theme_bw() +
    labs(title = "Distribution of Spinal Conditions in the Data",
         y = "Number of Images",
         x = "Spinal Condition") +
    scale_y_continuous(labels = scales::comma) +
    scale_x_discrete(labels = scales::label_wrap(20)) +
    theme(
        text = element_text(size = 16),
    )

ggsave("figures/data-analysis/condition_distribution.png", width = 8, height = 6, dpi = 300)