library(data.table)
library(dplyr)
library(ggplot2)

# Bayesian multilevel results analysis

df <- fread("/Users/beegroup/Nextcloud/PhD-Benedetto/Bayesian/data/results/bayes_results.csv")
cluster_df <-  fread("/Users/beegroup/Nextcloud/PhD-Benedetto/Bayesian/data/results/cluster_length.csv")
metadata <- fread('/Users/beegroup/Nextcloud/PhD-Benedetto/Bayesian/data/building_data_genome_2/metadata.csv')

# Preprocessing
df$pp_better <- as.factor(ifelse(df$partial_pooling_cvrmse< df$no_pooling_cvrmse & df$partial_pooling_cvrmse < df$complete_pooling_cvrmse, 'yes', 'no'))
df$np_better <- as.factor(ifelse(df$no_pooling_cvrmse< df$partial_pooling_cvrmse & df$no_pooling_cvrmse < df$complete_pooling_cvrmse, 'yes', 'no'))
df$cp_better <- as.factor(ifelse(df$complete_pooling_cvrmse< df$partial_pooling_cvrmse & df$complete_pooling_cvrmse < df$partial_pooling_cvrmse, 'yes', 'no'))

df$best_model <- as.factor(ifelse(df$partial_pooling_cvrmse< df$no_pooling_cvrmse,
                                  ifelse(df$partial_pooling_cvrmse < df$complete_pooling_cvrmse, 'pp', 'cp'), 
                                  ifelse(df$no_pooling_cvrmse < df$complete_pooling_cvrmse, 'np', 'cp')))

ggplot(df) + geom_bar(aes(best_model))
ggplot(df) + geom_boxplot(aes(partial_pooling_cvrmse))

avg_pp_cvrmse <- mean(df[['partial_pooling_cvrmse']])
avg_np_cvrmse <- mean(df[['no_pooling_cvrmse']])
avg_cp_cvrmse <- mean(df[['complete_pooling_cvrmse']])

# The avg cvrmse show that there's outliers that should be investigated manually

ggplot(df) + geom_histogram(aes(partial_pooling_cvrmse))

# How many buildings have CV(RMSE) higher than 1?
sum(df$partial_pooling_cvrmse>1)

# Average CV(RMSE) without the outlier buildings

mean(data.matrix(df[df$partial_pooling_cvrmse<1, 'partial_pooling_cvrmse']))
mean(data.matrix(df[df$no_pooling_cvrmse<1, 'no_pooling_cvrmse']))

# Find bad performing buildings

ids <- df[df$partial_pooling_cvrmse>1 & df$partial_pooling_cvrmse<5, 'id']
# Let's check if the partial pooling model performance is linked with the cluster length

df_full <- df %>%
  inner_join(cluster_df, metadata, by = 'id') %>%
  inner_join(metadata, by = c('id' = 'building_id'))

df_full$small_clusters <- as.factor(ifelse(df_full$len_smallest<500, 'yes', 'no'))

ggplot(df_full) + geom_bar(aes(best_model, fill = small_clusters))
ggplot(df_full) + geom_bar(aes(best_coverage, fill = small_clusters))

# Let's see if we can link it to any metadata
df_full$site_id <- as.factor(df_full$site_id)
df_full$primaryspaceusage <-  as.factor(df_full$primaryspaceusage)
ggplot(df_full) + geom_bar(aes(best_model, fill = site_id))
ggplot(df_full) + geom_bar(aes(best_model, fill = primaryspaceusage))

# Coverage and length analysis

df$best_coverage <- as.factor(ifelse(df$partial_pooling_coverage< df$no_pooling_coverage,
                                  ifelse(df$partial_pooling_coverage < df$complete_pooling_coverage, 'pp', 'cp'), 
                                  ifelse(df$no_pooling_coverage < df$complete_pooling_coverage, 'np', 'cp')))

df$widest_bands <- as.factor(ifelse(df$partial_pooling_length< df$no_pooling_length,
                                     ifelse(df$partial_pooling_length < df$complete_pooling_length, 'pp', 'cp'), 
                                     ifelse(df$no_pooling_length < df$complete_pooling_length, 'np', 'cp')))


ggplot(df) + geom_bar(aes(best_coverage))
ggplot(df) + geom_bar(aes(widest_bands))


# lost_ids <- mapply(setdiff, unique(cluster_df[,'id']), unique(df[, 'id']))
# lost_ids <- c(lost_ids, mapply(setdiff, unique(df[, 'id']), unique(cluster_df[,'id'])))

