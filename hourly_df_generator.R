#Data cleaning and analysis eplus_runner
rm (list=ls())
setwd("~/GitHub/GAM_Measurement_and_Verification")
source("functions.R")
library(dplyr)
library(tidyr)
library(reshape)
library(oce)
library(lubridate)
library(readr)
library(mgcv)
library(pracma)
library(pastecs)
library(data.table)
library(GA)
library(ggplot2)
library(extrafont)
library(plotly)
library(grid)
library(gridExtra)
library(rjson)
library(mgcViz)
library(mclust)
library(cluster)
library(clusterCrit)
library(hydroGOF)
library(parallel)
library(mlr)
library(xgboost)
library(glmnet)
library(padr)
library(caret)

#Select case and number of min/max clusters
case="cluster_as_variable"=list("k"=2:7, "modtype"="cluster_as_variable")
training_data_start_date <- "2016-01-01"

#Data import
heat_source <- 'electricity'
building = 'office'
tz_local = 'Europe/Rome'

setwd(paste0("~/Nextcloud//PhD-Benedetto/Energy_plus_simulations/idf_files/eppy/phase2/", building))

lat <- fromJSON(file = paste0(building,"_json.json"))$location[1]
long <- fromJSON(file = paste0(building,"_json.json"))$location[2]
measures <- fromJSON(file = paste0(building,"_json.json"))$measures
measures_by_type <- classify_measures(measures)
cdf <- data.frame(import_df(output_path = paste0(building,"_output/"), heating_source= heat_source))
cdf$dw <- as.factor(cdf$dw)
cdf$local_date <- as.Date(cdf$t,tz=tz_local)

cdf$total_electricity_ma <- rollapply(cdf$total_electricity, width = 3, align = 'center', partial = TRUE,
                                      FUN = mean)
plot(cdf$t,cdf$total_electricity)

###
# Clustering of the load curves when no EEM is applied
###

#Filter by date to check performance with less training data

cdf <- cdf[cdf$date >= training_data_start_date,]

#Filter by sd to eliminate the flat days (we'll manually add them after the clustering)
cdf_agg_sd <- cdf %>%
  group_by(local_date) %>%
  summarize(sd(total_electricity))

plot(cdf_agg_sd)
mean(cdf_agg_sd$`sd(total_electricity)`, na.rm = T)
quantile(cdf_agg_sd$`sd(total_electricity)`, na.rm =T)

#Hardcoded value to filter the sds
dates_sd_filtered <- cdf_agg_sd[cdf_agg_sd$`sd(total_electricity)`>6, 'local_date']
cdf_input_clust <- cdf[cdf$local_date %in% dates_sd_filtered$local_date & cdf$t<measures[[1]]$date,]

#Clustering
clustering <- clustering_load_curves(
  df = cdf_input_clust,
  tz_local = tz_local,
  time_column = "t",
  value_column = "total_electricity",
  temperature_column = "outdoor_temp",
  k=case$k,
  perc_cons = T,
  n_dayparts = 24,
  norm_specs = NULL,
  input_vars = c("load_curves", "daily_cons"), # POSSIBLE INPUTS: c("load_curves", "days_weekend", "days_of_the_week", "daily_cons", "daily_temp"),
  centroids_plot_file = "clustering.pdf",
  bic_plot_file = "bic.pdf",
  latex_font = F, 
  plot_n_centroids_per_row=4,
  minimum_days_for_a_cluster = 10)
clustering$df <- clustering$df[order(clustering$df$time),]
#ggplotly(ggplot(clustering$df[clustering$df$s!="NA",]) + geom_line(aes(time,value,col=s,group=1)))

###
# Classification of the load curves for the whole consumption time series
###

classification <- classifier_load_curves(
  df = cdf,
  df_centroids = clustering$centroids[,!(colnames(clustering$centroids) %in% c("s"))],
  clustering_mod = clustering$mod,
  tz_local=tz_local, 
  time_column = "t", 
  value_column = "total_electricity",
  temperature_column = "outdoor_temp",
  perc_cons = clustering$perc_cons,
  n_dayparts = clustering$n_dayparts, 
  norm_specs = clustering$norm_specs,
  input_vars = clustering$input_vars,
  plot_file = "classification.pdf",
  plot_as_output = T,
  latex_font = T,
  plot_n_centroids_per_row = 4
)

classification$df$cluster <- classification$df$shape1
cdf <- merge(cdf[,!(grepl("cluster",colnames(cdf)))],classification$df[,c("day","cluster")],by.x = "local_date",
             by.y = "day", all.x=T)

#Download and merge weather dataframe
wdf <- read_csv(paste0("meteo_data/", lat, "_", long, "_hist_hourly.csv" ))

df<-merge(cdf[,!(colnames(cdf) %in% colnames(wdf)[!grepl("time",colnames(wdf))])],wdf,by.x="t",by.y="time")

#Lines to export the df to apply towt in python (change indexes depending if we want measure 1 or measure 2)

# towt_meter_data <- cdf[8785:nrow(cdf),c('t','total_electricity')] %>%
#   dplyr::rename(start = t,  value = total_electricity )
# 
# write_csv(x = towt_meter_data, path = paste0('/Users/beegroup/ownCloud/PhD-Benedetto/Energy_plus_simulations/towt_analysis_files/',
#                                              building, '_meter_m2.csv'))
# 
# towt_temp_data <- wdf[8786:26305,c('time','temperature')] %>%
#   dplyr::rename(dt = time,  tempF = temperature )
# 
# write_csv(x = towt_temp_data, path = paste0('/Users/beegroup/ownCloud/PhD-Benedetto/Energy_plus_simulations/towt_analysis_files/',
#                                             building, '_temp_m2.csv'))


###
# Classifier model for predicting the load curve using the calendar data
###
# Variables hardcoded at this point
df$cluster_train <- df$cluster
df$cluster_pred <- df$cluster
df$dep <- 1

# Aggregation to daily
df_d <- daily_aggregation(df, heating_source = heat_source)
df_d <- df_d[,!(colnames(df_d) %in% c("cluster_train","cluster_pred"))]

# Measures applied in this building
dummydfs <- dummy_variables(df,df_d,measures)
df_d <- dummydfs$df_d
df <- dummydfs$df
gam_features <- dummydfs$gam_features

# Set dummy variables as factors
df_d$m <- as.factor(df_d$m)

for (measure in 1:length(measures)){
  df_d[,paste0('m',measure)] <- as.factor(df_d[,paste0('m',measure)])
}

#Visualize timeseries with measures applied

ggsave("timseries_by_measure.pdf", ggplot(df) + geom_line(aes(t,total_electricity, col = m)) + theme_bw() +
         
         theme(axis.text=element_text(size=18),axis.text.x = element_text( vjust = 0.5, margin = margin(t=5, r=0, b = 10, l =0)), 
               axis.text.y = element_text(margin = margin(t=0, r= 10, b = 0, l = 10)),
               axis.title=element_text(size=25))+labs(x="Date", y="Total energy consumption [kWh]") +
         
         theme(text= element_text(size=14, family="CM Roman"), axis.text = element_text(size=14, family="CM Roman")) +
         
         theme(
           legend.position = c(.95, .95),
           legend.justification = c("right", "top"),
           legend.box.just = "right",
           legend.margin = margin(6, 6, 6, 6), 
           legend.key.size = unit(1.3, "cm")
         ),
       width = 35, height = 20, units = "cm"
) 

#Visualize timeseries by cluster

ggsave('timseries_by_cluster_circle.pdf', ggplot(df_d[1:365,]) + geom_point(aes(t,total_electricity, col = cluster), 
                                                                            shape = 19, size= 3, alpha = 0.7) +
         theme_bw() + theme(text= element_text(size=20, family="CM Roman")) + 
         theme(
           legend.position = c(.95, .97),
           legend.justification = c("right", "top"),
           legend.box.just = "right",
           legend.margin = margin(6, 6, 6, 6)
         ) +
         theme(axis.text=element_text(size=18),axis.text.x = element_text( vjust = 0.5, margin = margin(t=5, r=0, b = 10, l =0)), 
               axis.text.y = element_text(margin = margin(t=0, r= 10, b = 0, l = 10)),
               axis.title=element_text(size=25))+labs(x="Date", y="Daily total electricity consumption [kWh]") + 
         guides(col=guide_legend(ncol=2)), 
       width = 35, height = 20, units = "cm")


ggplot(df_d[1:365,]) + geom_line(aes(t,total_electricity), col = df_d[1:365,'cluster']) +  theme_bw()

#Visualize cluster by day of the week

ggplot(df_d) + geom_bar(aes(dw,fill = cluster)) + theme_bw() 
ggplot(df_d) + geom_bar(aes(cluster,fill = dw)) + theme_bw()

## Input features for XGBoost model

# Add weekend info
df_d$weekend <- as.factor(ifelse(df_d$dw %in% c(6,7),1,0))

#Add holiday variable
holiday_list <- as.Date(c("2016-01-01", "2016-03-24", "2016-03-25", "2016-03-28", "2016-12-26", "2017-04-13",
                          "2017-04-14", "2017-04-17", "2017-12-25", "2017-12-26", "2018-01-01", "2018-03-29",
                          "2018-03-30", "2018-04-02", "2018-12-24", "2018-12-25", "2018-12-26"))

holiday_var <- rep(0,nrow(df_d))
df_d$holiday <- holiday_var
df_d[df_d$t %in% holiday_list, 'holiday'] <- 1
df_d$holiday <- as.factor(df_d$holiday)

# Add a continuous variable marking the day of the year using Fourier's transform
df_d$yearday <- as.numeric(strftime(df_d$t, format = "%j", tz= "UTC" ))
fourier_year <- do.call(cbind,fs(df_d$yearday/366, nharmonics = 2))
colnames(fourier_year) <- paste0('dayyear_', colnames(fourier_year))

# Add a continuous variable marking the day of the week using Fourier's transform
df_d$dw <- as.numeric(df_d$dw)
fourier_dw <- do.call(cbind,fs(df_d$dw/7, nharmonics = 2))
colnames(fourier_dw) <- paste0('dayweek_', colnames(fourier_dw))

# Add the cluster of yesterday
df_d$cluster_l1_ <- shift(df_d$cluster,1)
df_d$cluster_l7_ <- shift(df_d$cluster,7)
options(na.action='na.pass')
cluster_lagged <- as.data.frame(model.matrix(~0+cluster_l1_+cluster_l7_, df_d ))
df_d <- df_d[!(colnames(df_d) %in% c("cluster_l1_","cluster_l7_"))]

# Add the temperature and GHI splines
temp_spline <- as.data.frame(splines::bs(df_d$temperature,knots=3,degree=2))
colnames(temp_spline) <- paste0("temperature_sp_",colnames(temp_spline))
GHI_spline <- splines::bs(df_d$GHI,knots=3,degree=2)
colnames(GHI_spline) <- paste0("GHI_sp_",colnames(GHI_spline))
total_electricity_spline <- splines::bs(df_d$total_electricity,knots=5,degree=2)
colnames(total_electricity_spline) <- paste0("total_electricity_sp_",colnames(total_electricity_spline))

# Add all the terms in df_d
df_d <- cbind(df_d, fourier_year, fourier_dw, cluster_lagged, temp_spline, GHI_spline)

#Predict cluster_train and cluster_pred based on: dw, daily temp aggregate, month of year (continuous), holiday
df_d$cluster_train <- NULL
df_d$cluster_pred <- NULL
res <- predict_cluster(df_d)
cluster_pred <- df_d$cluster_pred <- res$df$cluster_pred
cluster_train <- df_d$cluster_train <- res$df$cluster_train
levels(df_d$cluster_train) <- c(levels(df_d$cluster_train),levels(df_d$cluster)[!(levels(df_d$cluster) %in% levels(df_d$cluster_train))])
xgb.importance(model = res$mod$`0`$mod$mod$learner.model)
caret::confusionMatrix(df_d$cluster,df_d$cluster_train)
df_d2 <- df_d
df_d2$cluster_train <- df_d2$cluster 

#Plot accuracy of Xgboost prediction
ggplot()+geom_point(aes(t, total_electricity,col=ifelse(cluster==cluster_train,"Good class.","Bad class.")),alpha=0.6,data=df_d)+
  geom_point(aes(t,total_electricity, shape="4"),alpha=0.6, data=df_d2[df_d$cluster!=df_d$cluster_train,])+
  scale_shape_discrete("Real",labels="Missclassified") + ylab("") + 
  facet_wrap(~cluster_train,ncol=1) + theme_bw() + theme(legend.box.just = "top") + scale_color_discrete("Prediction")


###
# Regression model for HVAC dependence
###

df <- merge(df[,!(colnames(df) %in% c("cluster_train","cluster_pred"))],df_d[,c("t","cluster_train","cluster_pred")],by.x="date",by.y="t")

# Temperature low-pass filter
df$temperature <- lp_vector(df$temperature,0.1)

#Analyse weather dependence and calculate optimized balance temperature
df$season <- ifelse(month(df$t) %in% c(11,12,1,2,3,4), 'winter', 'summer')
results_hvac <- add_dep_hvac(df = df, value_column= 'total_electricity', temperature_column = "temperature", 
                             ts_from_train = training_data_start_date, ts_to_train = measures[[1]]$date,
                             cluster_column = "cluster_train", time_column="t", tz_local= tz_local,
                             plot_file = "results_%s.pdf", season_column ="season", sig_lvl=0.05)

#Calculate temperature, sun altitude and wind speed vectors depending on the weather dependence
df$dep <- results_hvac$df$dep_hvac
Tbalance <- results_hvac$params[1]
hist <- results_hvac$params[2]
df$temp_dep_h <- ifelse(df$temperature > (Tbalance - hist) , 0, (Tbalance - hist - df$temperature)) * df$dep
df$temp_dep_c <- ifelse(df$temperature < (Tbalance + hist), 0, (df$temperature - Tbalance + hist)) * df$dep
df$GHI_dep <- df$GHI * df$dep
df$windSpeed_dep <- df$windSpeed * df$dep
df$cluster_hour <- as.integer(as.factor(paste0(df$cluster, df$hour)))
df_ <- select(df, t, total_electricity, hour, GHI_dep, temp_dep_c, temp_dep_h, temperature, cluster_hour, cluster, windSpeed_dep, m)
write.csv(df_, file = "/Users/beegroup/Github/Bayes-M&V/data/hourly_multilevel_office.csv")
