"""
Object for storing, loading and modifying results.
"""
import json
import copy
import os
import pickle

from .defaults import DEFAULTS

def load_json(file_name, output_dir):
    with open(os.path.join(output_dir, file_name) + '.json', "r") as f:
        d = json.load(f)

    return d

def save_json(file_name, results, output_dir):
    with open(os.path.join(output_dir, file_name) + '.json', "w") as f:
        json.dump(results, f, indent=4, sort_keys=True)
    
    return

def save_pkl(file_name, results, output_dir):
    with open(os.path.join(output_dir, file_name) + ".pkl", "wb") as f:
        pickle.dump(results, f)
    return


def load_pkl(file_name, output_dir):
    with open(os.path.join(output_dir, file_name) + ".pkl", "rb") as f:
        d = pickle.load(f)
    return d

class Container:
    def __init__(self, name: str, output_dir: str=DEFAULTS.output_dir, as_json: bool=True):
        self.name = name
        self.output_dir = output_dir
        self.as_json = as_json
        
        self.data = []
               
        if (not os.path.exists(self.output_dir)) and (output_dir != ''):
            os.mkdir(self.output_dir)
        
    def load(self):
        if self.as_json:
            self.data = load_json(self.name, self.output_dir)
        else:
            self.data = load_pkl(self.name, self.output_dir)
        return self
    
    def append(self, res: dict):
        self.data.append(res)
        return self
    
    def store(self):
        if self.as_json:
            save_json(self.name, self.data, self.output_dir)
        else:
            save_pkl(self.name, self.data, self.output_dir)
        return
        
        

