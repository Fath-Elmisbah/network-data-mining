import pandas as pd
import numpy as np
from netmine.clustering.clustering import SOMClustering, SOMBasedClusetring, WardClustering, KmeansClustering, Clustering, AgglomerativeClustering, KMeans
from netmine.clustering.state_io import json_load, json_store
import dill

path = 'Ericsson NW.csv'
date_column_idx = 0
identifying_columns_idxs = [1,2]
KPIs_columns_idxs = [3,4,5]
number_of_KPIs = len(KPIs_columns_idxs)
na_values = {KPIs_columns_idxs[0]:'#DIV/0',KPIs_columns_idxs[1]:'#DIV/0', KPIs_columns_idxs[2]:'#DIV/0'}
footer_rows_idxs = list(range(20296,20307))
datetime_format = '%m/%d/%Y'
columns_ac_names = ['AVAIL', 'CSSR', 'CDR']

data = pd.read_csv('Ericsson NW.csv', header=0, skiprows=footer_rows_idxs)

b = pd.DataFrame({'a':[1,2,3,4], 'b':['a','b','x','f']})
d = pd.DataFrame({'g':[1,2,3,4], 'b':['f','b','x','f']})
print(b.join(d))
