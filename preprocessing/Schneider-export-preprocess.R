setwd("/Users/beegroup/Nextcloud/PhD-Benedetto/Schneider-Forecasting/")
library(dplyr)
library(xts)
library(data.table)
library(mclust)
library(ggplot2)
library(tidyverse)
library(tidyr)
library(parallel)
library(pracma)
library(padr)
library(zoo)
library(GA)
library(penalized)
library(plotly)
library(lubridate)

# Data import ----

cdf <- fread("train.csv") %>% filter(SiteId ==50)
wdf <- fread("weather.csv") %>% filter(SiteId ==50)

cdf$Timestamp <- as.POSIXct(cdf$Timestamp, format= "%Y-%m-%d %H:%M:%OS")
wdf$Timestamp <- as.POSIXct(wdf$Timestamp, format= "%Y-%m-%d %H:%M:%OS")

ggplotly(ggplot(cdf) + geom_line(aes(Timestamp,Value)))
# Consumption Hourly aggregation, since we have data gaps we need to do the mean and multiply by 4
cdf$metered_hour <- floor_date(as.POSIXct(cdf$Timestamp, format= "%Y-%m-%d %H:%M:%OS") - seconds(1), "1 hour")
cdf_h <- data.frame(
  V = aggregate(cdf$Value,list("t"=cdf$metered_hour),FUN=mean)
)
cdf_h$V.x <- cdf_h$V.x*4
colnames(cdf_h) <- c("Timestamp", "Value")

# Temp hourly aggregation (can change to use rollmean and include all relevant data)
wdf$metered_hour <- floor_date(as.POSIXct(wdf$Timestamp, format= "%Y-%m-%d %H:%M:%OS"), "1 hour")
wdf_h <- data.frame(
  Temp = aggregate(wdf$Temperature,list("t"=wdf$metered_hour),FUN=mean)
)
colnames(wdf_h) <- c("Timestamp", "Temperature")

df_h <- merge(cdf_h, wdf_h, by = "Timestamp")
ggplot(df_h) + geom_line(aes(Timestamp, Value)) + geom_line(aes(Timestamp,Temperature*1000), col = 'red')
ggplot(df_h) + geom_jitter(aes(Temperature, Value))
write.csv(df_h, "/Users/beegroup/Github/Bayes-M&V/data/Id50_raw.csv")

# Preprocessing ----

setwd("/Users/beegroup/Github/Bayes-M&V")
source("functions.R")

id = 'multilevel_hourly'
head(df_h)
colnames(df_h) <- c("t","total_electricity", "outdoor_temp")
df_h$t <- as.POSIXct(df_h$t,tz="Europe/Madrid")
ggplotly(ggplot(df_h) + geom_line(aes(t,total_electricity)))

clustering <- clustering_load_curves(
  df = df_h,
  tz_local = "Europe/Madrid",
  time_column = "t", value_column = "total_electricity", temperature_column = "outdoor_temp",
  k=2:6,
  perc_cons = T,
  n_dayparts = 24,
  norm_specs = NULL,
  input_vars = c("daily_cons","daily_temp"), # POSSIBLE INPUTS: c("load_curves", "days_weekend", "days_of_the_week", "daily_cons", "daily_temp"),
  centroids_plot_file = "clustering.pdf",
  bic_plot_file = "bic.pdf",
  # centroids_plot_file = NULL,
  # bic_plot_file = NULL,#"bic.pdf",
  latex_font = F,
  plot_n_centroids_per_row=2,
  minimum_days_for_a_cluster = 10,
  force_plain_cluster = F,
  filename_prefix=paste(id,sep="~")
)

df_centroids <- clustering$centroids

# Classification of load patterns
classification <- classifier_load_curves(
  df = df_h,
  df_centroids = df_centroids[,!(colnames(df_centroids) %in% c("s"))],
  clustering_mod = setNames(lapply(names(clustering),function(i)clustering[[i]][["mod"]]),names(clustering)),
  tz_local = "Europe/Madrid",
  time_column = "t",
  value_column = "total_electricity",
  temperature_column = "outdoor_temp",
  perc_cons = clustering$perc_cons,
  n_dayparts = clustering$n_dayparts,
  norm_specs = clustering$norm_specs,
  input_vars = clustering$input_vars,
  plot_n_centroids_per_row = 2,
  # plot_file = NULL,
  plot_file = "classification.pdf",
  filename_prefix=paste(id,sep="~")
)

df_centroids <- reshape2::melt(df_centroids,id_vars=c("s"))
colnames(df_centroids) <- c("s","dayhour","value")
df_centroids_avg <- clustering$centroids_avg

classification <- classification[,c("date","s")]
classification <- classification[!duplicated(classification),]
classification_from_clustering <- clustering[["classified"]]
colnames(classification_from_clustering) <- c("date","s")
classification_from_clustering$s <- as.numeric(classification_from_clustering$s)
classification <- rbind(classification_from_clustering[is.finite(classification_from_clustering$s),],classification)
classification <- classification[!duplicated(classification[,c("date")]),]
classification$s <- sprintf("%02i",classification$s)
df_centroids_count <- as.data.frame(table(classification$s))
colnames(df_centroids_count) <- c("cluster","days")

df_h$t <- as.POSIXct(df_h$t, tz="Europe/Madrid")
df_h$local_date <- as.Date(df_h$t,tz="Europe/Madrid")
df_h <- merge(df_h,classification[,c("date","s")],by.x="local_date",by.y="date",all.x=T)
df_h$s<-as.factor(df_h$s)

df_h$all <- "all"
df_h$GHI <- 0
df_h$windSpeed <- 0
characterization <- characterizer(
  df_ini = df_h,
  tz_local = "Europe/Madrid",
  time_column = "t",
  value_column = "total_electricity",
  temperature_column = "outdoor_temp",
  GHI_column = "GHI",
  intercept_column = "weekday",
  windSpeed_column = "windSpeed",
  group_column ="all",
  hours_of_each_daypart = 4,
  centroids = df_centroids,
  centroids_summary = df_centroids_avg,
  classification = classification[,c("date","s")]
)

ggplot(characterization$df) +
  geom_point(aes(outdoor_temp,total_electricity,col=s), size=0.4) +
  #geom_point(aes(outdoor_temp,pred), color="red", size=0.1) +
  ylab(bquote("W/m"^2)) + xlab(bquote("Temperature ["*degree*"C]")) +
  theme_bw() +
  theme(
    legend.position = "none",
    text=element_text(size=14),
    strip.text.y.right = element_text(angle = 0),
    # strip.placement.y = "inside",
    # strip.text.y = element_text(angle = 180),
    strip.background = element_blank(),
    axis.text.x = element_text(angle=60,hjust = 1))

characterization$df$t <- as.POSIXct(characterization$df$t) 
ggplotly(ggplot(characterization$df) +
           geom_line(aes(t,total_electricity)) +
           geom_line(aes(t,pred), color="red",alpha=0.5) +
           ylab("electricity [kWh]") + xlab("temperature [ºC]") +
           theme_bw())

df_export <- characterization$df %>% select(t, total_electricity, outdoor_temp, s, outdoor_temp_h, outdoor_temp_c, 
                                            starts_with("daypart_fs"))

write.csv(df_export, file = "data/Id50_preprocessed.csv")
