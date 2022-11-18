"""
Object for storing, loading and modifying results.
"""
import json
import copy
import os

def load_json(file_name, output_dir):
    with open(output_dir + file_name + '.json', "r") as f:
        d = json.load(f)

    return d

def save_json(file_name, results, output_dir):
    with open(output_dir + file_name + '.json', "w") as f:
        json.dump(results, f, indent=4, sort_keys=True)
    
    return

class Container:
    def __init__(self, name: str, output_dir: str='output/'):
        self.name = name
        self.output_dir = output_dir
        self.data = []
        
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        
    def load(self):
        self.data = load_json(self.name, self.output_dir)
        return self
    
    def append(self, res: dict):
        self.data.append(res)
        return self
    
    def store(self):
        save_json(self.name, self.data, self.output_dir)
        return
        
        

