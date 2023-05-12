# -*- coding: utf-8 -*-
"""
Created on Fri May 12 12:46:01 2023

@author: Srikanth
"""

import numpy as np    
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.preprocessing import StandardScaler 
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation, DBSCAN
from sklearn.mixture import GaussianMixture 
from scipy.optimize import curve_fit

# To use this experimental feature, we need to explicitly ask for it:
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics.cluster import silhouette_score, silhouette_samples, calinski_harabasz_score, davies_bouldin_score

#Importing warnings so that it may ignore warnings
import warnings
warnings.filterwarnings('ignore')

# To use this experimental feature, we need to explicitly ask for it:
from sklearn.experimental import enable_iterative_imputer  # noqa

# Importing warnings so that it may ignore warnings
warnings.filterwarnings('ignore')


def plot_cluster(x, y, data, centers=None, title=None, c=None):
    """ Plot a scatter plot of clustered data with optional cluster centers.

    Args:
        x, y : str
            Columns in `data` to be plotted.
        data : pandas.DataFrame
            DataFrame containing the data to be plotted.
        centers : pandas.DataFrame, optional
            DataFrame containing the cluster centers. Each row represents a cluster center.
            The DataFrame should have the same columns as `data`.
        title : str, optional
            Title for the plot.
        c : str, optional
            Column name in `data` to use for coloring the points based on clusters.

    Returns:
        axes : matplotlib Axes
            The Axes object with the plot drawn onto it.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    if c:
        scatter = ax.scatter(data[x], data[y], c=data[c], cmap='viridis')
        plt.colorbar(scatter, ax=ax, label=c)
    else:
        ax.scatter(data[x], data[y], color='gray')

    if centers is not None:
        ax.scatter(
            centers[x],
            centers[y],
            marker='x',
            color='red',
            s=100,
            label='Cluster Center')

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.legend()

    plt.show()
    return ax


# Define the RS variable with a random state value
RS = np.random.RandomState(42)  # You can use any desired random state value


def plot_boxolin(x, y, data, title=None):
    """ Plot a box plot and a violin plot.

    Args:
        x, y : str
            Columns in `data` to be plotted. x is the 'groupby' attribute.
        data : pandas.DataFrame
            DataFrame containing `x` and `y` columns
        title : str, optional
            Title for the plots.

    Returns:
        axes : matplotlib Axes
            The Axes object with the plot drawn onto it.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    sns.boxplot(x=x, y=y, data=data, ax=axes[0])
    sns.violinplot(x=x, y=y, data=data, scale='area', ax=axes[1])

    axes[0].set_xlabel(x)  # Set x-axis label for the box plot
    axes[0].set_ylabel(y)  # Set y-axis label for the box plot
    axes[1].set_xlabel(x)  # Set x-axis label for the violin plot
    axes[1].set_ylabel(y)  # Set y-axis label for the violin plot

    if title:
        fig.suptitle(title)  # Set the title of the plots
    plt.show()
    return axes


def cluster(model, X, **kwargs):
    """ Run a clustering model and return predictions.

    Args:
        model : {sklearn.cluster, sklearn.mixture, or hdbscan}
            Model to fit and predict
        X : pandas.DataFrame
            Data used to fit `model`
        **kwargs : `model`.fit_predict() args, optional
            Keyword arguments to be passed into `model`.fit_predict()
    Returns:
        (labels,centers) : tuple(array, pandas.DataFrame)
            A tuple containing cluster labels and a DataFrame of cluster centers formated with X columns
    """
    clust_labels = model.fit_predict(X, **kwargs)
    centers = X.assign(**{model.__class__.__name__: clust_labels}  # assign a temp column to X with model name
                       ).groupby(model.__class__.__name__, sort=True).mean()  # group on temp, gather mean of labels
    return (clust_labels, centers)


def score_clusters(X, labels):
    """ Calculate silhouette, calinski-harabasz, and davies-bouldin scores

    Args:
        X : array-like, shape (``n_samples``, ``n_features``)
            List of ``n_features``-dimensional data points. Each row corresponds
            to a single data point.

        labels : array-like, shape (``n_samples``,)
            Predicted labels for each sample.
    Returns:
        scores : dict
            Dictionary containing the three metric scores
    """
    scores = {'silhouette': silhouette_score(X, labels),
              'calinski_harabasz': calinski_harabasz_score(X, labels),
              'davies_bouldin': davies_bouldin_score(X, labels)
              }
    return scores


def plot_curve_fit(x, y, data, title=''):
    """Plot a curve fit of the given data.

    Args:
        x, y : str
            names of variables in ``data``
        data : pandas.DataFrame
            desired plotting data
        title : str, optional
            title of plot

    Returns:
        ax : matplotlib Axes
            the Axes object with the plot drawn onto it.
    """

    def func(x, a, b, c):
        """Function to fit the data."""
        return a * np.exp(-b * x) + c

    # Perform the curve fit
    popt, _ = curve_fit(func, data[x], data[y], p0=[1, 1, 1])

    # Generate points for the fitted curve
    x_fit = np.linspace(data[x].min(), data[x].max(), 100)
    y_fit = func(x_fit, *popt)

    # Plot the original data and the fitted curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(data[x], data[y], label='Data')
    ax.plot(x_fit, y_fit, 'r-', label='Curve Fit')

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend()
    ax.set_title(title)
    return ax


pd.set_option("display.max_columns", 30)  # Increase columns shown
whr = pd.read_excel('E:\\Chapter2OnlineData.xls')
print(whr.columns)
full_colnames = [
    'Country',
    'Year',
    'Life_Ladder',
    'Log_GDP',
    'Social_support',
    'Life_Expectancy',
    'Freedom',
    'Generosity',
    'Corruption_Perception',
    'Positive_affect',
    'Negative_affect',
    'Confidence_natGovt',
    'Democratic_Quality',
    'Delivery_Quality',
    'sdLadder',
    'cvLadder',
    'giniIncWB',
    'giniIncWBavg',
    'giniIncGallup',
    'trust_Gallup',
    'trust_WVS81_84',
    'trust_WVS89_93',
    'trust_WVS94_98',
    'trust_WVS99_2004',
    'trust_WVS2005_09',
    'trust_WVS2010_14']
core_col = full_colnames[:9]
ext_col = full_colnames[:14] + full_colnames[17:19]
whr.columns = full_colnames
whr.columns = whr.columns.str.replace(
    'Most people can be trusted',
    'trust_in_people')
whr.columns = whr.columns.str.replace(' ', '_')
whr.columns = whr.columns.str.replace('[(),]', '')  # Strip parens and commas
whr.columns
# print(whr.iloc[np.r_[0:3,-3:0]])
whr_ext = whr[ext_col].copy()
whr_ext.groupby('Country').Year.count().describe()
# print(whr_ext)
# Get latest year indices
latest_idx = whr_ext.groupby('Country').Year.idxmax()
whrl = whr_ext.iloc[latest_idx].set_index('Country')

# Check NAs in the core data set
print(whrl[whrl[core_col[1:]].isna().any(axis=1)])
imputer = IterativeImputer(estimator=BayesianRidge(
), random_state=RS, max_iter=15).fit(whr_ext.iloc[:, 1:])
whrl_imp = pd.DataFrame(
    imputer.transform(whrl),
    columns=whrl.columns,
    index=whrl.index)
# Impute on latest forward filled data
whrffl_imp = pd.DataFrame(
    imputer.transform(whrl),
    columns=whrl.columns,
    index=whrl.index)
ss = StandardScaler()
whrX = pd.DataFrame(
    ss.fit_transform(
        whrffl_imp.drop(
            columns='Year')), columns=whrffl_imp.drop(
                columns='Year').columns, index=whrffl_imp.index)
print(whrX.head())
whr_grps = whrX.copy()

distortions = []
for n in range(2, 10):
    model = KMeans(n_clusters=n, random_state=RS).fit(whrX)
    distortions.append(model.inertia_)
    labs = model.labels_
    score = score_clusters(whrX, labs)
    # Uncomment the line below to print the evaluation metrics
    # print(f'n_clusters: {n}\n', score)

# KMeans clusters
km = KMeans(n_clusters=3, random_state=RS)
clabels_km, cent_km = cluster(km, whrX)
whr_grps['KMeans'] = clabels_km
cent_km

# Plot each K-Means cluster separately
for cluster_label in set(clabels_km):
    cluster_data = whr_grps[whr_grps['KMeans'] == cluster_label]
    plot_cluster(
        'Log_GDP',
        'Corruption_Perception',
        cluster_data,
        centers=cent_km,
        title=f'K-Means Cluster {cluster_label}',
        c='KMeans')

# DBSCAN
db = DBSCAN(eps=0.3)
clabels_db, cent_db = cluster(db, whrX)
whr_grps['DBSCAN'] = clabels_db
cent_db

# Plot each DBSCAN cluster separately
for cluster_label in set(clabels_db):
    cluster_data = whr_grps[whr_grps['DBSCAN'] == cluster_label]
    plot_cluster(
        'Log_GDP',
        'Corruption_Perception',
        cluster_data,
        centers=cent_db,
        title=f'DBSCAN Cluster {cluster_label}',
        c='DBSCAN')

# plot_boxolin
plot_boxolin('KMeans', 'Log_GDP', whr_grps, title='Box Plot and Violin Plot')

# Curve fit plot
plot_curve_fit(
    'Log_GDP',
    'Corruption_Perception',
    whr_grps,
    title='Curve Fit Plot')