import os
import json
import numpy as np
import pickle
import dill
from types import FunctionType

# todo: use dill instead
def json_store(filepath, state, filename='state.json'):
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    for k,v in state.items():
        if k.startswith('-'):
            _, type, name = k.split('-')
            if type == 'dill':
                state[k] = name + '.pkl'
                with open(filepath + state[k], 'wb') as file:
                    dill.dump(v,file)
            elif type == 'recursive':
                json_store(filepath + name + '\\', v, filename)
    with open(filepath + filename,'w') as file:
        json.dump(state,file, indent=4)

def json_load(filepath, filename='state.json'):
    with open(filepath + filename,'r') as file:
        json_data = json.load(file)
    for k,v in json_data.items():
        if k.startswith('-'):
            _, type, name = k.split('-')
            if type == 'dill':
                name = v
                with open(filepath + name, 'rb') as file:
                    json_data[k] = dill.load(file)
            elif type == 'recursive':
                json_data[k] = json_load(filepath + name + r'\\', filename)
    return json_data

def store_dill(object, filename):
    filepath, filename = os.path.split(filename)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    with open(filepath + '\\' + filename, 'wb') as file:
        dill.dump(object, file)

def load_dill(filename):
    with open(filename, 'rb') as file:
        object = dill.load(file)
    return object