import tqdm
import numpy as np
import torch
import copy
import time
import datetime

from torch.utils.data import DataLoader

from .datasets.main import get_dataset, infer_shapes
from .models.main import get_model
from .optim.main import get_optimizer, get_scheduler
from .metrics import get_metric_function

from .utils import l2_norm, grad_norm

class Base:
    def __init__(self, name: str, config: dict, device: str):
        self.name = name
        self.config = copy.deepcopy(config)
        
        if torch.cuda.is_available() and device=='cuda':
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        
        self.data_path = 'data/'
        self.seed = 1234567
        self.run_seed = 456789 + config.get('run_id', 0)
        
        self.check_config()
        
    def check_config(self):
        
        # create defaults for missing config keys
        if 'batch_size' not in self.config.keys():
            self.config['batch_size'] = 1
        
        if 'dataset_kwargs' not in self.config.keys():
            self.config['dataset_kwargs'] = dict()

        if 'model_kwargs' not in self.config.keys():
            self.config['model_kwargs'] = dict()
        
        # check necessary config keys
        for k in ['loss_func', 'score_func', 'opt']:
            assert k in self.config.keys(), f"You need to specify {k} in the config file."

        return
    
    def setup(self):
        
        #============ Data =================
        self.train_set = get_dataset(config=self.config, split='train', seed=self.seed, path=self.data_path)
        self.val_set = get_dataset(config=self.config, split='val', seed=self.run_seed, path=self.data_path)
        
        self.config['_input_dim'], self.config['_output_dim'] = infer_shapes(self.train_set)
        
        # construct train loader
        _gen = torch.Generator()
        _gen.manual_seed(self.run_seed)
        self.train_loader = DataLoader(self.train_set, drop_last=True, shuffle=True, generator=_gen,
                                       batch_size=self.config['batch_size'])
               
        #============ Model ================
        torch.manual_seed(self.seed) # Reseed to have same initialization   
        torch.cuda.manual_seed_all(self.seed)        
        
        self.model = get_model(self.config)
        self.model.to(self.device)
        print(self.model)
        
        #============ Optimizer ============
        opt_obj, hyperp = get_optimizer(self.config['opt'])
        self.opt = opt_obj(params=self.model.parameters(), **hyperp)       
        if not hasattr(self.opt, 'state'):
            self.opt['state'] = dict()
        
        print(self.opt)
        self.sched = get_scheduler(self.config['opt'], self.opt)
        
        #============ Results ==============
        self.results = {'config': self.config,
                        'history': {},
                        'summary': {}}
        
        return 
    
    def run(self):
        start_time = str(datetime.datetime.now())
        score_list = []
    
        for epoch in range(self.config['max_epoch']):
            
            # Record metrics
            score_dict = {'epoch': epoch}
            score_dict['lr'] = self.sched.get_last_lr()[0] # must be stored before sched.step()
                           
            # Train one epoch
            s_time = time.time()
            self.train_epoch()
            e_time = time.time()
        
            # Validation
            with torch.no_grad():
                metric_dict = {'loss': self.config['loss_func'], 'score': self.config['score_func']}
                train_dict = self.validate(self.train_set, 
                                           metric_dict = metric_dict,
                                           )  
            
                val_dict = self.validate(self.val_set, 
                                         metric_dict = metric_dict,
                                         )                     
                       
                # Record more metrics
                score_dict.update(train_dict)
                score_dict.update(val_dict)
                score_dict['train_epoch_time'] = e_time - s_time       
                score_dict['model_norm'] = l2_norm(self.model)        
                score_dict['grad_norm'] = self.opt.state.get('grad_norm')
                
                # Add score_dict to score_list
                score_list += [score_dict]
        
        end_time = str(datetime.datetime.now())
        
        # ==== store =====================
        self.results['history'] = copy.deepcopy(score_list)
        self.results['summary']['start_time'] = start_time
        self.results['summary']['end_time'] = end_time
        return

    def train_epoch(self):
        """
        Train one epoch.
        """
        loss_function = get_metric_function(self.config['loss_func'])
                
        self.model.train()
        pbar = tqdm.tqdm(self.train_loader)
        
        for batch in pbar:
            self.opt.zero_grad()    
            
            # get batch and compute model output
            data, targets = batch['data'].to(device=self.device), batch['targets'].to(device=self.device)
            out = self.model(data)
                   
            closure = lambda: loss_function(out, targets, backwards=True)
            loss_val = self.opt.step(closure)
            
            pbar.set_description(f'Training - {loss_val:.3f}')
        
        # store gradient norm
        self.opt.state['grad_norm'] = grad_norm(self.opt)
        print("Current learning rate", self.sched.get_last_lr()[0])
        
        # update learning rate             
        self.sched.step()       

        return
    
    def validate(self, dataset, metric_dict):
        """
        Validate model for a given dataset (train or val), and for several metrics.
        
        metric_dict:
            Should have the form {'metric_name1': metric1, 'metric_name2': metric2, ...}
        """
        # create temporary DataLoader
        dl = torch.utils.data.DataLoader(dataset, drop_last=True, 
                                         batch_size=self.config['batch_size'])
        pbar = tqdm.tqdm(dl)
        
        self.model.eval()
        score_dict = dict(zip(metric_dict.keys(), np.zeros(len(metric_dict))))
        
        for batch in pbar:
            # get batch and compute model output
            data, targets = batch['data'].to(device=self.device), batch['targets'].to(device=self.device)
            out = self.model(data)
            
            for _met, _met_fun in metric_dict.items():
                this_metric = get_metric_function(_met_fun)
                # metric takes average over batch ==> multiply with batch size
                score_dict[_met] += this_metric(out, targets).item() * data.shape[0] 
                
            pbar.set_description(f'Validating {dataset.split}.')
        
        
        for _met in metric_dict.keys():
            # Get from sum to average
            score_dict[_met] = float(score_dict[_met] / len(dl.dataset))
            # add split in front of names
            score_dict[dataset.split + '_' + _met] = score_dict.pop(_met)
        
        return score_dict

    
    

        



