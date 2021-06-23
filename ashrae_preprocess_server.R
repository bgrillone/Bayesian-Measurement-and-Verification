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
library(kernlab)
library(igraph)
library(nnet)
setwd("/root/benedetto/Bayesian-Measurement-and-Verification")
source("preprocessing/functions_updated.R")

a <- commandArgs(trailingOnly = T)
df <- read.csv(paste0("/root/benedetto/results/buildings/", a[1],".csv"), stringsAsFactors = F)
#df <- read.csv("/Users/beegroup/Nextcloud/PhD-Benedetto/Bayesian/data/debugging/Crow_education_Keisha.csv", stringsAsFactors = F)

id = 'multilevel_hourly'
df$t <- as.POSIXct(df$t,tz="Europe/Madrid", format = "%Y-%m-%d %H:%M:%S")
df <- df[complete.cases(df$t),]
print(getwd())
clustering <- clustering_load_curves(
  df = df[df$t<as.POSIXct("2017-01-01"),],
  tz_local = "Europe/Madrid",
  time_column = "t",
  value_column = "total_electricity",
  temperature_column = "outdoor_temp",
  perc_cons = T,
  kmax=30,
  n_dayparts = 24,
  norm_specs = NULL,
  input_vars = c("load_curves"), # POSSIBLE INPUTS: c("load_curves", "days_weekend", "days_of_the_week", "daily_cons", "daily_temp"),
  centroids_plot_file = NULL,#paste0(a[1], "_clustering.pdf"),
  plot_n_centroids_per_row=3,
  filename_prefix="",
  folder_plots="/root/benedetto/results/plots/clustering_plots/"
)

df_centroids <- reshape2::melt(clustering$centroids,id_vars=c("s"))
colnames(df_centroids) <- c("s","dayhour","value")
df_centroids_avg <- clustering$centroids_avg

# Classification of load patterns

classification <- classifier_load_curves(
  df = df[df$t>=as.POSIXct("2017-01-01"),],
  df_centroids = clustering$centroids,
  clustering_centroids = clustering$clustering_centroids,
  clustering_mod_calendar = clustering$mod_calendar,
  tz_local = "Europe/Madrid",
  time_column = "t",
  value_column = "total_electricity",
  temperature_column = "outdoor_temp",
  perc_cons = clustering$perc_cons,
  n_dayparts = clustering$n_dayparts,
  norm_specs = clustering$norm_specs,
  input_vars = clustering$input_vars,
  plot_n_centroids_per_row = 3,
  # plot_file = NULL,
  plot_file = NULL,#paste0(a[1], "_classification.pdf"),
  filename_prefix="",
  folder_plots= "/root/benedetto/results/plots/clustering_plots/"
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
classification$s <- sprintf("%02i",as.integer(classification$s))
df_centroids_count <- as.data.frame(table(classification$s))
colnames(df_centroids_count) <- c("cluster","days")

df$t <- as.POSIXct(df$t, tz="Europe/Madrid")
df$local_date <- as.Date(df$t,tz="Europe/Madrid")
df <- merge(df,classification[,c("date","s")],by.x="local_date",by.y="date",all.x=T)
df$s<-as.factor(df$s)

# Add Fourier terms
df <- add_fs_daypart(df, 't')

# Add daypart and weekday
hours_of_each_daypart = 4
df$daypart <- as.factor(floor(hour(df[,'t'])/hours_of_each_daypart))
df$weekday <- strftime(df[,'local_date'], "%u")

# Export
df_export <- df %>% select(t, total_electricity, outdoor_temp, s, daypart,
                                            weekday, starts_with("daypart_fs"))

df_export <- df_export[complete.cases(df_export),]

write.csv(df_export,paste0("/root/benedetto/results/buildings/", a[1],"_preprocess.csv"), row.names = F)


# df$all <- "all"
# df$GHI <- 0
# df$windSpeed <- 0
# df <- df[order(df$t),]
# df <- df[!is.na(df$s),]
# 
# characterization <- characterizer(
#   df_ini = df,
#   tz_local = "Europe/Madrid",
#   time_column = "t",
#   value_column = "total_electricity",
#   temperature_column = "outdoor_temp",
#   GHI_column = "GHI",
#   intercept_column = "weekday",
#   windSpeed_column = "windSpeed",
#   date_column = "local_date",
#   group_column ="all",
#   hours_of_each_daypart = 4,
#   centroids = df_centroids,
#   centroids_summary = df_centroids_avg,
#   classification = classification[,c("date","s")]
# )
# 
# #indicators <- indicators_estimator(characterization, meteo_df = df[,c("t","outdoor_temp","windSpeed","GHI")])
# 
# ggplot(characterization$df) +
#   geom_point(aes(outdoor_temp,total_electricity), size=0.4) +
#   geom_point(aes(outdoor_temp,pred), color="red", size=0.1) +
#   facet_wrap(~s)+
#   ylab(bquote("W/m"^2)) + xlab(bquote("Temperature ["*degree*"C]")) +
#   theme_bw() +
#   theme(
#     legend.position = "none",
#     text=element_text(size=14),
#     strip.text.y.right = element_text(angle = 0),
#     # strip.placement.y = "inside",
#     # strip.text.y = element_text(angle = 180),
#     strip.background = element_blank(),
#     axis.text.x = element_text(angle=60,hjust = 1))
# 
# characterization$df$t <- as.POSIXct(characterization$df$t) 
# ggplot(characterization$df) +
#   geom_line(aes(t,total_electricity)) +
#   geom_line(aes(t,pred), color="red",alpha=0.5) +
#   ylab("electricity [kWh]") + xlab("temperature [ºC]") +
#   theme_bw()
# 
# df_export <- characterization$df %>% select(t, total_electricity, outdoor_temp, s, daypart,
#                                             weekday, outdoor_temp_lp_h, outdoor_temp_lp_c,
#                                             outdoor_temp_h, outdoor_temp_c, starts_with("daypart_fs"))
# 
# df_export <- df_export[complete.cases(df_export),]
# 
# write.csv(df_export,paste0("/root/benedetto/results/buildings/", a[1],"_preprocess.csv"), row.names = F)
# 
# n_clusters <- length(levels(df$s))
# cluster_length <- df %>% 
#   group_by(s) %>%
#   summarise(no_rows = length(s)) %>%
#   drop_na()
# 
# hours_smallest_cluster <- min(cluster_length$no_rows)
# tot_hours <- nrow(df_export)
# 
# cluster_df <- data.frame(n_cluster = n_clusters,
#                          len_smallest = hours_smallest_cluster,
#                          ts_length = tot_hours,
#                          id = a[2])
# 
# if (file.exists("/root/benedetto/results/cluster_length.csv")){
#   dat = fread("/root/benedetto/results/cluster_length.csv", header = T)
#   final_export <- rbind(dat, cluster_df)
# }else{
#   final_export <- cluster_df
# }
# 
# write.csv(final_export, "/root/benedetto/results/cluster_length.csv", row.names = F)
# 
