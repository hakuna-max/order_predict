# load packages
library(tidyverse)
library(ggthemes)
library(ggridges)
library(hrbrthemes)
library(viridis)

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
  geom_point(aes(color = n), alpha = 0.3, shape = 16, size = 5) +
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


# Region and Quantity -----------------------------------------------------

data |>
  ggplot(mapping = aes(
    x = log(ord_qty),
    y = sales_region_code,
    fill = ..x..
  )) +
  geom_density_ridges_gradient(scale = 3, rel_min_height = 0.01) +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_discrete(expand = c(0, 0)) +
  scale_fill_viridis(name = "Log Order Quantity", option = "E") +
  coord_cartesian(clip = "off") +
  labs(
    title = "The Distribution of Log Order Quantity in Regions",
    x = "Log order Quantity",
    y = "Regions"
  ) +
  theme_ipsum() +
  theme(
    legend.position = "none",
    panel.spacing = unit(0.1, "lines"),
    strip.text.x = element_text(size = 8)
  )

data |>
  filter(item_price < 5000 & ord_qty < 2000) |>
  group_by(item_price) |> 
  ggplot(mapping = aes(
    x = item_price,
    y = log(ord_qty)
  )) +
  geom_point(alpha = 0.1,
  color = "#f0650e") +
  geom_smooth() + # gam
  facet_wrap(~sales_region_code) +
  labs(
    title = "The Relationship between Order Quantity and Price in different Sale Regions",
    x = "Price",
    y = "Log Order Quantity"
  )


# sales channels ----------------------------------------------------------

data |> 
  ggplot(mapping = aes(
    x = sales_chan_name,
    y = log(ord_qty)
  )) +
  geom_boxplot()


data |> 
  mutate(item_price = round(item_price)) |> 
  group_by(item_price) |> 
  filter(item_price < 5000 & ord_qty < 2000) |> 
  ggplot(mapping = aes(
    x = item_price,
    y = log(ord_qty)
  )) +
  geom_point(alpha = 0.1) +
  geom_smooth() +
  facet_wrap(~sales_chan_name) +
  labs(
    title = "The Relationship between Order Quantity and Price related to different Sale Channels"
  )
