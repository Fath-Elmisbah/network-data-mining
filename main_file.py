import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.metrics import silhouette_score
from netmine.clustering.state_io import store_dill, load_dill

from netmine.clustering.preprocessing import IgnoreNanValuesHandler, SigmoidWith10pecrentScaler
from netmine.clustering.clustering import SOMClustering, KmeansClustering, WardClustering,\
    SOMBasedClusetring
from netmine.clustering.postprocessing import order_clusters
from netmine.clustering.optimization import Optimizer
from netmine.clustering.visualization import plot_dataset, plot_clustering,\
    plot_clusters_features_histograms, plot_som_map, plot_som_map_mapping
import output

## parameters
# input
path = 'Ericsson NW.csv'
date_column_idx = 0
identifying_columns_idxs = [1,2]
KPIs_columns_idxs = [3,4,5]
number_of_KPIs = len(KPIs_columns_idxs)
na_values = {KPIs_columns_idxs[0]:'#DIV/0',KPIs_columns_idxs[1]:'#DIV/0', KPIs_columns_idxs[2]:'#DIV/0'}
footer_rows_idxs = list(range(20296,20307))
datetime_format = '%m/%d/%Y'
columns_ac_names = ['AVAIL', 'CSSR', 'CDR']

# preprocessing
a10 = np.array([60,60,2])
a90 = np.array([95,95,.1])
# clustering
num_of_clusters_range = range(2,15)
som_width_range = range(7,19,2)
som_hight_top_limit = 19
som_num_of_iters1 = 10000
som_num_of_iters2 = 200000
kmeans_num_of_iters = 10000
kmeans_num_of_trials = 20

# visualization and output
nbest = 10
output_path = 'clusters.csv'
opt_path = 'clusterings_compared.txt'
opt_state_path = 'optimization\\state_1.pkl'
opt_restore_path = 'optimization\\state.pkl'

# flow control (for debugging only)
debug_input = True
debug_preprocessing = True
debug_clustering = False
debug_restore = True
debug_labels_calculations = True
debug_visualization = False
debug_output = False
debug_trials = True


################################################
## input
if debug_input:
    dataset = pd.read_csv(path, header=0, na_values=na_values, skiprows=footer_rows_idxs)

    dataset.iloc[:,date_column_idx] = pd.to_datetime(dataset.iloc[:,date_column_idx], format=datetime_format)

    for KPI_column_idx in KPIs_columns_idxs:
      dataset.iloc[:,KPI_column_idx] = pd.to_numeric(dataset.iloc[:, KPI_column_idx])

    columns = dataset.iloc[:, KPIs_columns_idxs].columns
    dataset = dataset.iloc[:, KPIs_columns_idxs].values


################################################
## preprocessing
if debug_preprocessing:
    nan_handler = IgnoreNanValuesHandler()
    dataset = nan_handler.fit_handle(dataset)

    sigmoid_scaler = SigmoidWith10pecrentScaler(a10=a10, a90=a90)
    dataset = sigmoid_scaler.fit_handle(dataset)


################################################
## clustering
if debug_clustering:
    clusterings = []
    for num_of_clusters in num_of_clusters_range:
        for width in som_width_range:
            for hight in range(width,som_hight_top_limit,2):
                som_clustering1 = SOMClustering(width=width,hight=hight, num_of_iters1=som_num_of_iters1, num_of_iters2=som_num_of_iters2)
                som_clustering2 = SOMClustering(width=width, hight=hight, num_of_iters1=som_num_of_iters1, num_of_iters2=som_num_of_iters2)
                ward_clustering = WardClustering(num_of_clusters=num_of_clusters)
                k_means_clustering = KmeansClustering(num_of_clusters=num_of_clusters, num_of_iters=kmeans_num_of_iters, num_of_trials=kmeans_num_of_trials)
                som_ward_clustering = SOMBasedClusetring(som_clustering1,ward_clustering)
                som_k_means_clustering = SOMBasedClusetring(som_clustering2,k_means_clustering)
                clusterings.append(som_ward_clustering)
                clusterings.append(som_k_means_clustering)
    clustering_optimizer = Optimizer(clusterings,
                                     criteria=lambda x,y: 1*silhouette_score(y, x.fit_predict(y)), store=store_dill)
    clustering_optimizer.set_criteria_parameters(dataset)
    clustering_optimizer.optimize(store_filename=opt_state_path, verbose=True)


####################################
## restore
if debug_restore:
    clustering_optimizer = Optimizer.from_io(None, load_dill, opt_restore_path)


####################################
## labels calculations
if debug_labels_calculations:
    best_n = clustering_optimizer.accepted_choices[-1:-1 * nbest - 1:-1]
    labels = []
    for clustering in best_n:
        labels.append(clustering.predict(dataset))
    labels = np.array(labels).T


################################################
## visualization
if debug_visualization:
    dataset = sigmoid_scaler.inverse_handle(dataset)
    plot_dataset(dataset, columns, 'dataset')
    for i in range(nbest):
        plot_clustering(dataset, labels[:,i], columns)
        som_labels = best_n[i].som_clustering.predict(dataset)
        plot_som_map_mapping(labels[:,i], som_labels, best_n[i].som_clustering.som)

    plt.show()


################################################
## output
if debug_output:
    labels = order_clusters(labels)
    dataset = pd.DataFrame(dataset, columns= columns)

    output.output_clusterings(dataset, labels, output_path)
    output.output_optimization(clustering_optimizer, opt_path, n_best=nbest)


#################################################
## trials
if debug_trials:
    model = KmeansClustering(2)
    print(silhouette_score(dataset, clustering_optimizer.accepted_choices[-1].predict(dataset)))
    print(silhouette_score(dataset, model.fit_predict(dataset)))
