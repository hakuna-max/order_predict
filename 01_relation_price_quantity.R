# load packages
library(tidyverse)
library(ggthemes)

# load data
data <- read.csv("data/order_train0.csv")

# transform data
data <- as_tibble(data)


cols <- c(
  "sales_region_code", "item_code", "first_cate_code",
  "second_cate_code", "sales_chan_name"
)

data <- data |>
  mutate_at(cols, factor) |>
  mutate(order_date = ymd(order_date))

glimpse(data)


# Price and Quantity ------------------------------------------------------

# visualization of the relationship between price and quantity
data |>
  mutate(item_price = round(item_price)) |>
  group_by(item_price) |>
  summarise(
    aver_ord_qty = mean(ord_qty),
    n = n(),
  ) |>
  ggplot(mapping = aes(
    x = item_price,
    y = aver_ord_qty,
  )) +
  geom_point(aes(color = n),  alpha = 0.3, shape = 16, size = 5) +
  geom_smooth(method = "loess", color = "black") +
  labs(
    title = "The Relationship Between Price and Order Quantity",
    x = "Price",
    y = "Average Order Quantity",
    color = "No. of Each Price"
  ) +
  coord_cartesian(xlim = c(100, 5000), ylim = c(0, 300)) +
  theme_economist_white() +
  scale_color_gradient(low = "#0091ff", high = "#f0650e")

# ggsave("01_relationship_price_quantity.png", path = "results/")


# Region and Qauntity -----------------------------------------------------





