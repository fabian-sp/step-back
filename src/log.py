"""
Object for storing, loading and modifying results.
"""
import json
import copy


def load_json(file_name):
    with open('output/' + file_name + '.json', "r") as f:
        d = json.load(f)

    return d

def save_json(file_name, results):
    with open('output/' + file_name + '.json', "w") as f:
        json.dump(results, f, indent=4, sort_keys=True)
    
    return

class Container:
    def __init__(self, name: str):
        self.name = name
        self.data = []
        
    def load(self):
        self.data = load_json(self.name)
        return self
    
    def append(self, res: dict):
        self.data.append(res)
        return self
    
    def store(self):
        save_json(self.name, self.data)
        return
        
        

