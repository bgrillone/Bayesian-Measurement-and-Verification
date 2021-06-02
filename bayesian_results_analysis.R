library(data.table)
library(dplyr)
library(ggplot2)

# Bayesian multilevel results analysis

df <- fread("/Users/beegroup/Github/Bayes-M&V/data/results/bayes_results.csv")
cluster_df <-  fread("/Users/beegroup/Github/Bayes-M&V/data/results/cluster_length.csv")
metadata <- fread('/Users/beegroup/Github/Bayes-M&V/data/building_data_genome_2/metadata.csv')

df <- df %>%
  inner_join(cluster_df, metadata, by = 'id') %>%
  inner_join(metadata, by = c('id' = 'building_id'))

# Some observations are getting lost because they are not contained in either of the dataframes:
# should check manually what's going on
# Get the ids that we're losing

lost_ids <- mapply(setdiff, unique(cluster_df[,'id']), unique(df[, 'id']))
lost_ids <- c(lost_ids, mapply(setdiff, unique(df[, 'id']), unique(cluster_df[,'id'])))


df$pp_better <- as.factor(ifelse(df$partial_pooling_cvrmse< df$no_pooling_cvrmse & df$partial_pooling_cvrmse < df$complete_pooling_cvrmse, 'yes', 'no'))
df$np_better <- as.factor(ifelse(df$no_pooling_cvrmse< df$partial_pooling_cvrmse & df$no_pooling_cvrmse < df$complete_pooling_cvrmse, 'yes', 'no'))
df$cp_better <- as.factor(ifelse(df$complete_pooling_cvrmse< df$partial_pooling_cvrmse & df$complete_pooling_cvrmse < df$partial_pooling_cvrmse, 'yes', 'no'))
df$best_model <- as.factor(ifelse(df$partial_pooling_cvrmse< df$no_pooling_cvrmse,
                                  ifelse(df$partial_pooling_cvrmse < df$complete_pooling_cvrmse, 'pp',
                                         ifelse(df$no_pooling_cvrmse < df$complete_pooling_cvrmse, 'np', 'cp'))))

avg_pp_cvrmse <- mean(df[['partial_pooling_cvrmse']])
avg_np_cvrmse <- mean(df[['no_pooling_cvrmse']])
avg_cp_cvrmse <- mean(df[['complete_pooling_cvrmse']])

# The avg cvrmse show that there's outliers that should be investigated manually

df$best_model <- as.factor(ifelse(df$partial_pooling_cvrmse< df$no_pooling_cvrmse,
                                  ifelse(df$partial_pooling_cvrmse < df$complete_pooling_cvrmse, 'pp', 'cp'), 
                                  ifelse(df$no_pooling_cvrmse < df$complete_pooling_cvrmse, 'np', 'cp')))
                                         
                                         
ggplot(df) + geom_point(aes(sqm, partial_pooling_cvrmse, col = best_model)) + ylim(c(0,1))

summary(df$best_model)

# Is the cluster length helping us in understanding if partial pooling will be better?
# 
