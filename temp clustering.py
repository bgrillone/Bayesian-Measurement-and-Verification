import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.cm as cm
from bokeh.layouts import gridplot
from bokeh.plotting import figure, output_file, show
from sklearn import mixture
from sklearn.metrics import silhouette_score, silhouette_samples

df = pd.read_csv("/Users/beegroup/Github/Bayes-M&V/data/Id50_preprocessed.csv")

# steps of this algorithm
# instead of excluding the days from the flat cluster, we should exclude the days with no temperature
# dependence (== days with temp_dep_h around zero)
# run gmm clustering on heating temp and cooling temp separately
# use different metrics to automatically choose the optimal number of clusters
# Plot clusters and contour plots
# Assign points to clusters

df_heat = df[df.outdoor_temp_h >0].filter(['outdoor_temp_h', 'total_electricity'], axis = 1)
df_cool = df[df.outdoor_temp_c >0].filter(['outdoor_temp_c', 'total_electricity'], axis = 1)

#plot temperature dependence
#Cons vs temp
p1 = figure(plot_width=800, plot_height=400)
p2 = figure(plot_width=800, plot_height=400)
# add a circle renderer with a size, color, and alpha
p1.circle(df_heat.outdoor_temp_h, df_heat.total_electricity, size=5, color="navy", alpha=0.5)
p2.circle(df_cool.outdoor_temp_c, df_cool.total_electricity, size=5, color="navy", alpha=0.5)
# show the results
show(gridplot([p1,p2], ncols = 2))

# There's two pretty clear temperature dependence clusters for both heating and cooling
# How can we detect them automatically? Clustering or quantile regression? Let's try both approaches

#Gaussian mixture model
range_n_clusters = [2,3,4]

#heating case
for n_clusters in range_n_clusters:

    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.5, 0.5])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(df_heat) + (n_clusters + 1) * 10])

    # run the clustering
    gmm = mixture.GaussianMixture(n_components=n_clusters, covariance_type='full')
    gmm.fit(df_heat)
    cluster_labels = gmm.fit_predict(df_heat)
    silhouette_avg = silhouette_score(df_heat, cluster_labels)

    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    sample_silhouette_values = silhouette_samples(df_heat, cluster_labels)

    y_lower  = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(df_heat['outdoor_temp_h'],df_heat['total_electricity'], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')
    plt.show()


# Cooling case


for n_clusters in range_n_clusters:

    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.5, 0.5])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(df_cool) + (n_clusters + 1) * 10])

    # run the clustering
    gmm = mixture.GaussianMixture(n_components=n_clusters, covariance_type='full')
    gmm.fit(df_cool)
    cluster_labels = gmm.fit_predict(df_cool)
    silhouette_avg = silhouette_score(df_cool, cluster_labels)

    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    sample_silhouette_values = silhouette_samples(df_cool, cluster_labels)

    y_lower  = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(df_cool['outdoor_temp_c'],df_cool['total_electricity'], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')
    plt.show()

# Clustering seems to work pretty well in this case, the highest silhouette score is assigned to the case
# with 2 clusters. The problem when the number of clusters rises is that we only want to actually cluster along one
# direction. For this reason doing quantile regression may make more sense than GMM clustering

#Get a df with the point classification (hardcoded)

gmm = mixture.GaussianMixture(n_components=2, covariance_type='full')
gmm.fit(df_cool)
cluster_labels_cool = gmm.fit_predict(df_cool)
df_cool['temp_c_cluster'] = cluster_labels_cool

gmm.fit(df_heat)
cluster_labels_heat = gmm.fit_predict(df_heat)
df_heat['temp_h_cluster'] = cluster_labels_heat

df_ = pd.concat([df, df_heat.temp_h_cluster, df_cool.temp_c_cluster], axis = 1)
# add random cluster for na values so that the model is able to run (are we influencing the prediction?)
df_.temp_h_cluster = df_.temp_h_cluster.fillna(np.nanmax(df_.temp_h_cluster)+1)
df_.temp_c_cluster = df_.temp_c_cluster.fillna(np.nanmax(df_.temp_c_cluster)+1)
df_.temp_h_cluster =  df_.temp_h_cluster.astype('int64')
df_.temp_c_cluster =  df_.temp_c_cluster.astype('int64')
df_.to_csv('/Users/beegroup/Github/Bayes-M&V/data/Id50_preprocessed2.csv')
