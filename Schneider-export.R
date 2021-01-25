setwd("/Users/beegroup/Nextcloud/PhD-Benedetto/Schneider-Forecasting/")

library(data.table)
library(dplyr)
library(lubridate)
library(xts)
library(ggplot2)

#Load data from SideId 50
cdf <- fread("train.csv") %>% filter(SiteId ==50)
wdf <- fread("weather.csv") %>% filter(SiteId ==50)

cdf$Timestamp <- as.POSIXct(cdf$Timestamp, format= "%Y-%m-%d %H:%M:%OS")
wdf$Timestamp <- as.POSIXct(wdf$Timestamp, format= "%Y-%m-%d %H:%M:%OS")

# Consumption Hourly aggregation
cdf$metered_hour <- floor_date(as.POSIXct(cdf$Timestamp, format= "%Y-%m-%d %H:%M:%OS") - seconds(1), "1 hour")
cdf_h <- data.frame(
  V = aggregate(cdf$Value,list("t"=cdf$metered_hour),FUN=sum)
)
colnames(cdf_h) <- c("Timestamp", "Value")

# Temp hourly aggregation (might change to include temp values at :00 both for the previous and the following hour mean temp)
wdf$metered_hour <- floor_date(as.POSIXct(wdf$Timestamp, format= "%Y-%m-%d %H:%M:%OS"), "1 hour")
wdf_h <- data.frame(
  Temp = aggregate(wdf$Temperature,list("t"=wdf$metered_hour),FUN=mean)
)
colnames(wdf_h) <- c("Timestamp", "Temperature")

df_h <- merge(cdf_h, wdf_h, by = "Timestamp")
ggplot(df_h) + geom_line(aes(Timestamp, Value)) + geom_line(aes(Timestamp,Temperature*1000), col = 'red')
ggplot(df_h) + geom_jitter(aes(Temperature, Value))
write.csv(df_h, "/Users/beegroup/Github/Bayes-M&V/data/Id50_hourly.csv")

