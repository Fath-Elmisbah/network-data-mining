import numpy as np
import pandas as pd
from netmine.clustering.optimization import Optimizer

def output_clusterings(dataset, labels, path, names=None):
    dataset = dataset.copy()
    if len(labels.shape) == 1:
        labels.reshape((-1,1))
    if names == None:
        names = ['clustering ' + str(i) for i in range(labels.shape[1])]
    for i,name in enumerate(names):
        dataset[name] = labels[:,i]
    dataset.to_csv(path)

def output_optimization(optimization, path, n_best=3, head=''):
    top = optimization.accepted_choices[-1:-1*n_best-1:-1]
    top_scores = optimization.scores[-1:-1*n_best-1:-1]
    output = head + '\n'
    for i,accepted_choice in enumerate(top):
        output = output + str(i) + ' - ' + str(accepted_choice) + '\n score : ' + \
        str(top_scores[i]) + '\n'
    output = output + '\n'
    file = open(path,'w+')
    file.write(output)
    file.close()



