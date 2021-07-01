library(data.table)
library(dplyr)
library(ggplot2)
library(gridExtra)

# Bayesian multilevel results analysis

df_1 <- fread("/Users/beegroup/Nextcloud/PhD-Benedetto/Bayesian/data/results/bayes_results_iteration_1.csv")
df_2 <- fread("/Users/beegroup/Nextcloud/PhD-Benedetto/Bayesian/data/results/bayes_results_iteration_2.csv")
df_3 <- fread("/Users/beegroup/Nextcloud/PhD-Benedetto/Bayesian/data/results/bayes_results_iteration_3.csv")
df_5 <- fread("/Users/beegroup/Nextcloud/PhD-Benedetto/Bayesian/data/results/bayes_results_iteration_5.csv")
cluster_df <-  fread("/Users/beegroup/Nextcloud/PhD-Benedetto/Bayesian/data/results/cluster_length.csv")
metadata <- fread('/Users/benedetto/Nextcloud/PhD-Benedetto/Bayesian/data/building_data_genome_2/metadata.csv')

# Model comparison
df_1$pp_better <- as.factor(ifelse(df_1$partial_pooling_cvrmse< df_1$no_pooling_cvrmse & df_1$partial_pooling_cvrmse < df_1$complete_pooling_cvrmse, 'yes', 'no'))
df_1$np_better <- as.factor(ifelse(df_1$no_pooling_cvrmse< df_1$partial_pooling_cvrmse & df_1$no_pooling_cvrmse < df_1$complete_pooling_cvrmse, 'yes', 'no'))
df_1$cp_better <- as.factor(ifelse(df_1$complete_pooling_cvrmse< df_1$partial_pooling_cvrmse & df_1$complete_pooling_cvrmse < df_1$partial_pooling_cvrmse, 'yes', 'no'))

df_1$best_model <- as.factor(ifelse(df_1$partial_pooling_cvrmse< df_1$no_pooling_cvrmse,
                                  ifelse(df_1$partial_pooling_cvrmse < df_1$complete_pooling_cvrmse, 'pp', 'cp'), 
                                  ifelse(df_1$no_pooling_cvrmse < df_1$complete_pooling_cvrmse, 'np', 'cp')))

df_2$pp_better <- as.factor(ifelse(df_2$partial_pooling_cvrmse< df_2$no_pooling_cvrmse & df_2$partial_pooling_cvrmse < df_2$complete_pooling_cvrmse, 'yes', 'no'))
df_2$np_better <- as.factor(ifelse(df_2$no_pooling_cvrmse< df_2$partial_pooling_cvrmse & df_2$no_pooling_cvrmse < df_2$complete_pooling_cvrmse, 'yes', 'no'))
df_2$cp_better <- as.factor(ifelse(df_2$complete_pooling_cvrmse< df_2$partial_pooling_cvrmse & df_2$complete_pooling_cvrmse < df_2$partial_pooling_cvrmse, 'yes', 'no'))

df_2$best_model <- as.factor(ifelse(df_2$partial_pooling_cvrmse< df_2$no_pooling_cvrmse,
                                    ifelse(df_2$partial_pooling_cvrmse < df_2$complete_pooling_cvrmse, 'pp', 'cp'), 
                                    ifelse(df_2$no_pooling_cvrmse < df_2$complete_pooling_cvrmse, 'np', 'cp')))

grid.arrange(
  ggplot(df_1) + geom_bar(aes(best_model)),
  ggplot(df_2) + geom_bar (aes(best_model))
)
  
pp_box_1 <- ggplot(df_1) + geom_boxplot(aes(partial_pooling_cvrmse)) + xlim(c(0,0.5))
np_box_1 <- ggplot(df_1) + geom_boxplot(aes(no_pooling_cvrmse)) + xlim(c(0,0.5))
cp_box_1 <- ggplot(df_1) + geom_boxplot(aes(complete_pooling_cvrmse)) + xlim(c(0,0.5))
grid.arrange(pp_box_1, np_box_1, cp_box_1)

pp_box_2 <- ggplot(df_2) + geom_boxplot(aes(partial_pooling_cvrmse)) + xlim(c(0,0.5))
np_box_2 <- ggplot(df_2) + geom_boxplot(aes(no_pooling_cvrmse)) + xlim(c(0,0.5))
cp_box_2 <- ggplot(df_2) + geom_boxplot(aes(complete_pooling_cvrmse)) + xlim(c(0,0.5))
grid.arrange(pp_box_2, np_box_2, cp_box_2)

avg_pp_cvrmse_1 <- mean(df_1[['partial_pooling_cvrmse']])
avg_np_cvrmse_1 <- mean(df_1[['no_pooling_cvrmse']])
avg_cp_cvrmse_1 <- mean(df_1[['complete_pooling_cvrmse']])

avg_pp_cvrmse_2 <- mean(df_2[['partial_pooling_cvrmse']])
avg_np_cvrmse_2 <- mean(df_2[['no_pooling_cvrmse']])
avg_cp_cvrmse_2 <- mean(df_2[['complete_pooling_cvrmse']])

# The avg cvrmse show that there's outliers that should be investigated manually

# CVRMSE and NMBE analysis
# select the df that will be used in this analysis
df <- df_3
cvrmse_df <- df %>% 
  select(id, partial_pooling_cvrmse, no_pooling_cvrmse, complete_pooling_cvrmse) %>%
  melt(id.vars = 'id')

nmbe_df <- df %>% 
  select(id, partial_pooling_nmbe, no_pooling_nmbe, complete_pooling_nmbe) %>%
  melt(id.vars = 'id')

cov_df <- df %>% 
  select(id, partial_pooling_adj_coverage, no_pooling_adj_coverage, complete_pooling_adj_coverage) %>%
  melt(id.vars = 'id')
       
# Boxplots
cvrmse_box <- ggplot(cvrmse_df) + geom_boxplot(aes(variable, value, fill = variable)) + ylim(c(0,0.5)) + theme_bw() + 
  theme(legend.position = "none") + theme(axis.title.x=element_blank()) + 
  labs(y="CV(RMSE)") + scale_x_discrete(labels=c("partial_pooling_cvrmse" = "partial_pooling", 
                                                            "no_pooling_cvrmse" = "no_pooling",
                                                            "complete_pooling_cvrmse" = "complete_pooling"))  
  

nmbe_box <- ggplot(nmbe_df) + geom_boxplot(aes(variable, value, fill = variable)) + ylim(c(-1,1)) + theme_bw() + 
  theme(legend.position = "none") + theme(axis.title.x=element_blank()) + 
  labs(y="NMBE") + scale_x_discrete(labels=c("partial_pooling_nmbe" = "partial_pooling", 
                                                            "no_pooling_nmbe" = "no_pooling",
                                                            "complete_pooling_nmbe" = "complete_pooling"))

cov_box <- ggplot(cov_df) + geom_boxplot(aes(variable, value, fill = variable)) + ylim(c(1,2)) + theme_bw() + 
  theme(legend.position = "none") + theme(axis.title.x=element_blank()) + 
  labs(y="Adj Coverage") + scale_x_discrete(labels=c("partial_pooling_adj_coverage" = "partial_pooling", 
                                             "no_pooling_adj_coverage" = "no_pooling",
                                             "complete_pooling_adj_coverage" = "complete_pooling"))

cvrmse_violin <- ggplot(cvrmse_df) + geom_violin(aes(variable, value, fill = variable), scale = 'count') + ylim(c(0,0.5)) + theme_bw() + 
  theme(legend.position = "none") + theme(axis.title.x=element_blank()) + 
  labs(y="CV(RMSE)") + scale_x_discrete(labels=c("partial_pooling_cvrmse" = "partial_pooling", 
                                                            "no_pooling_cvrmse" = "no_pooling",
                                                            "complete_pooling_cvrmse" = "complete_pooling"))

nmbe_violin <- ggplot(nmbe_df) + geom_violin(aes(variable, value, fill = variable), scale = 'count') + ylim(c(-1,1)) + theme_bw() + 
  theme(legend.position = "none") + theme(axis.title.x=element_blank()) + 
  labs(y="NMBE") + scale_x_discrete(labels=c("partial_pooling_nmbe" = "partial_pooling", 
                                             "no_pooling_nmbe" = "no_pooling",
                                             "complete_pooling_nmbe" = "complete_pooling"))

cov_violin <- ggplot(cov_df) + geom_violin(aes(variable, value, fill = variable), scale = 'count') + ylim(c(1,1.5)) + theme_bw() + 
  theme(legend.position = "none") + theme(axis.title.x=element_blank()) + 
  labs(y="Adj Coverage") + scale_x_discrete(labels=c("partial_pooling_adj_coverage" = "partial_pooling", 
                                                     "no_pooling_adj_coverage" = "no_pooling",
                                                     "complete_pooling_adj_coverage" = "complete_pooling"))

ggsave(filename = 'Nextcloud/PhD-Benedetto/Bayesian/data/results/plots/error_box.pdf', 
       plot= grid.arrange(cvrmse_box, nmbe_box))
ggsave(filename = 'Nextcloud/PhD-Benedetto/Bayesian/data/results/plots/error_violin.pdf', 
       plot= grid.arrange(cvrmse_violin, nmbe_violin))
ggsave(filename = 'Nextcloud/PhD-Benedetto/Bayesian/data/results/plots/coverage_box.pdf', 
       plot= cov_box)
ggsave(filename = 'Nextcloud/PhD-Benedetto/Bayesian/data/results/plots/coverage_violin.pdf', 
       plot= cov_violin)

# Improvement from second best to best model 

df$pp_improvement <- ifelse(df$best_model == 'pp', ifelse(df$no_pooling_cvrmse < df$complete_pooling_cvrmse, 
                                                          df$no_pooling_cvrmse - df$partial_pooling_cvrmse,
                                                          df$complete_pooling_cvrmse - df$partial_pooling_cvrmse), NA)
df$np_improvement <- ifelse(df$best_model == 'np', ifelse(df$partial_pooling_cvrmse < df$complete_pooling_cvrmse, 
                                                          df$partial_pooling_cvrmse - df$no_pooling_cvrmse,
                                                          df$complete_pooling_cvrmse - df$no_pooling_cvrmse), NA)
mean(df$pp_improvement, na.rm= T)
mean(df$np_improvement, na.rm= T)
# This calculation is very sensible to outliers so it should be further verified

# Coverage and length analysis

df$best_coverage <- as.factor(ifelse(df$partial_pooling_coverage< df$no_pooling_coverage,
                                  ifelse(df$partial_pooling_coverage < df$complete_pooling_coverage, 'pp', 'cp'), 
                                  ifelse(df$no_pooling_coverage < df$complete_pooling_coverage, 'np', 'cp')))

df$widest_bands <- as.factor(ifelse(df$partial_pooling_length< df$no_pooling_length,
                                     ifelse(df$partial_pooling_length < df$complete_pooling_length, 'pp', 'cp'), 
                                     ifelse(df$no_pooling_length < df$complete_pooling_length, 'np', 'cp')))


ggplot(df) + geom_bar(aes(best_coverage))
ggplot(df) + geom_bar(aes(widest_bands))

# Let's check if the partial pooling model performance is linked with the cluster length

df_full <- df_2 %>%
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

# Outlier analysis 
# How many buildings have CV(RMSE) higher than 1?
sum(df$partial_pooling_cvrmse>1)

# Average CV(RMSE) without the outlier buildings

mean(data.matrix(df[df$partial_pooling_cvrmse<1, 'partial_pooling_cvrmse']))
mean(data.matrix(df[df$no_pooling_cvrmse<1, 'no_pooling_cvrmse']))

# Find bad performing buildings

ids <- df[df$partial_pooling_cvrmse>1 & df$partial_pooling_cvrmse<5, 'id']

# ADVI vs NUTS analysis
df_old <- fread("/Users/benedetto/Nextcloud/PhD-Benedetto/Bayesian/data/results/bayes_results_iteration_5.csv")

df$nuts_better <- as.factor(ifelse(df$nuts_binomial_cvrmse < df$advi_dep_cvrmse, 'yes', 'no'))
df$nuts_better_cov <- as.factor(ifelse(df$nuts_binomial_adjusted_coverage < df$advi_dep_adjusted_coverage, 'yes', 'no'))

table(df$nuts_better)
table(df$nuts_better_cov)

cvrmse_df <- df_old %>% 
  select(id, nuts_binomial_cvrmse, nuts_binomial_2_cvrmse, advi_dep_cvrmse, advi_dep_2_cvrmse) %>%
  melt(id.vars = 'id')

nmbe_df <- df_old %>% 
  select(id, nuts_binomial_nmbe, nuts_binomial_2_nmbe, advi_dep_nmbe, advi_dep_2_nmbe) %>%
  melt(id.vars = 'id')

cov_df <- df_old %>% 
  select(id, nuts_binomial_adjusted_coverage, nuts_binomial_2_adjusted_coverage, 
         advi_dep_adjusted_coverage, advi_dep_2_adjusted_coverage) %>%
  melt(id.vars = 'id')

# Boxplots
cvrmse_box <- ggplot(cvrmse_df) + geom_boxplot(aes(variable, value, fill = variable)) + ylim(c(0,1)) + theme_bw() + 
  theme(legend.position = "none") + theme(axis.title.x=element_blank()) + 
  labs(y="CV(RMSE)") + scale_x_discrete(labels=c("nuts_binomial_cvrmse" = "nuts", 
                                                 "nuts_binomial_2_cvrmse" = "nuts_2",
                                                 "advi_dep_cvrmse" = "advi",
                                                 "advi_dep_2_cvrmse" = "advi_2"))  


nmbe_box <- ggplot(nmbe_df) + geom_boxplot(aes(variable, value, fill = variable)) + ylim(c(-1,1)) + theme_bw() + 
  theme(legend.position = "none") + theme(axis.title.x=element_blank()) + 
  labs(y="NMBE") + scale_x_discrete(labels=c("nuts_binomial_cvrmse" = "nuts", 
                                             "nuts_binomial_2_cvrmse" = "nuts_2",
                                             "advi_dep_cvrmse" = "advi",
                                             "advi_dep_2_cvrmse" = "advi_2"))  

cov_box <- ggplot(cov_df) + geom_boxplot(aes(variable, value, fill = variable)) + ylim(c(1,10)) + theme_bw() + 
  theme(legend.position = "none") + theme(axis.title.x=element_blank()) + 
  labs(y="Adj Coverage") + scale_x_discrete(labels=c("nuts_binomial_adjusted_coverage" = "nuts", 
                                                     "nuts_binomial_2_adjusted_coverage" = "nuts_2",
                                                     "advi_dep_adjusted_coverage" = "advi",
                                                     "advi_dep_2_adjusted_coverage" = "advi_2"))  

cvrmse_violin <- ggplot(cvrmse_df) + geom_violin(aes(variable, value, fill = variable), scale = 'count') + ylim(c(0,0.5)) + theme_bw() + 
  theme(legend.position = "none") + theme(axis.title.x=element_blank()) + 
  labs(y="CV(RMSE)") + scale_x_discrete(labels=c("nuts_binomial_cvrmse" = "nuts", 
                                                 "nuts_binomial_2_cvrmse" = "nuts_2",
                                                 "advi_dep_cvrmse" = "advi",
                                                 "advi_dep_2_cvrmse" = "advi_2"))  

nmbe_violin <- ggplot(nmbe_df) + geom_violin(aes(variable, value, fill = variable), scale = 'count') + ylim(c(-1,1)) + theme_bw() + 
  theme(legend.position = "none") + theme(axis.title.x=element_blank()) + 
  labs(y="NMBE") + scale_x_discrete(labels=c("nuts_binomial_cvrmse" = "nuts", 
                                             "nuts_binomial_2_cvrmse" = "nuts_2",
                                             "advi_dep_cvrmse" = "advi",
                                             "advi_dep_2_cvrmse" = "advi_2"))  

cov_violin <- ggplot(cov_df) + geom_violin(aes(variable, value, fill = variable), scale = 'count') + ylim(c(1,10)) + theme_bw() + 
  theme(legend.position = "none") + theme(axis.title.x=element_blank()) + 
  labs(y="Adj Coverage") + scale_x_discrete(labels=c("nuts_binomial_cvrmse" = "nuts", 
                                                     "nuts_binomial_2_cvrmse" = "nuts_2",
                                                     "advi_dep_cvrmse" = "advi",
                                                     "advi_dep_2_cvrmse" = "advi_2"))  

ggsave(filename = '/Users/beegroup/Nextcloud/PhD-Benedetto/Bayesian/data/results/plots/error_box_it_5.pdf', 
       plot= grid.arrange(cvrmse_box, nmbe_box))
ggsave(filename = '/Users/beegroup/Nextcloud/PhD-Benedetto/Bayesian/data/results/plots/error_violin_it_5.pdf', 
       plot= grid.arrange(cvrmse_violin, nmbe_violin))
ggsave(filename = '/Users/beegroup/Nextcloud/PhD-Benedetto/Bayesian/data/results/plots/coverage_box_it_5.pdf', 
       plot= cov_box)
ggsave(filename = '/Users/beegroup/Nextcloud/PhD-Benedetto/Bayesian/data/results/plots/coverage_violin_it_5.pdf', 
       plot= cov_violin)

# Calculate the improvement between NUTS and ADVI
df$nuts_improvement <- ifelse(df$nuts_better == 'yes', df$advi_dep_cvrmse - df$nuts_binomial_cvrmse, NA)

df$advi_improvement <- ifelse(df$nuts_better == 'no', df$nuts_binomial_cvrmse - df$advi_dep_cvrmse, NA)

mean(df[df$nuts_binomial_cvrmse<1, ]$nuts_improvement, na.rm= T)
mean(df[df$advi_dep_cvrmse<1, ]$advi_improvement, na.rm= T)

# Import iteration 6 (ADVI with uniform prior on temperatures 10-25 )
df <- fread("Nextcloud/PhD-Benedetto/Bayesian/data/results/bayes_results_iteration_6.csv")

advi_cvrmse <- ggplot(df) + geom_boxplot(aes(advi_dep_cvrmse)) + xlim(0,0.6) + coord_flip() + theme_bw()
advi_cov <- ggplot(df) + geom_boxplot(aes(advi_dep_adjusted_coverage)) + xlim(1, 4) + coord_flip() + theme_bw()

grid.arrange(cvrmse_box, advi_cvrmse, ncol = 2 )
grid.arrange(cov_box, advi_cov, ncol = 2)

df_merged <- left_join(df_old, df, by = 'id')

ggplot(df_merged) + geom_boxplot(aes(advi_dep_adjusted_coverage.x))+ xlim(1,2.5) + coord_flip() + theme_bw()

grid.arrange(cvrmse_box,
             ggplot(df_merged) + geom_boxplot(aes(advi_dep_cvrmse.y))+ xlim(0,1) + coord_flip() + theme_bw(), ncol = 2)

grid.arrange(cov_box,
             ggplot(df_merged) + geom_boxplot(aes(advi_dep_adjusted_coverage.y))+ xlim(1,10) + coord_flip() + theme_bw(), ncol = 2)

# Which buildings were not calculated?

ids <- metadata[!metadata$building_id %in% df$id & metadata$electricity == 'Yes', 'building_id']
