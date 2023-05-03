import tqdm
import numpy as np
import torch
import copy
import time
import datetime
import warnings

from torch.utils.data import DataLoader

from .datasets.main import get_dataset, infer_shapes
from .models.main import get_model
from .optim.main import get_optimizer, get_scheduler
from .metrics import Loss

from .utils import l2_norm, grad_norm, ridge_opt_value, logreg_opt_value

class Base:
    def __init__(self, name: str, config: dict, device: str='cpu', data_dir: str='data/'):
        self.name = name
        self.config = copy.deepcopy(config)
        self.data_dir = data_dir

        print("CUDA available? ", torch.cuda.is_available())
        
        if torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device('cpu')
            
        print("Using device: ", self.device)
        
        self.seed = 1234567
        self.run_seed = 456789 + config.get('run_id', 0)
        torch.backends.cudnn.benchmark = False
        # see https://pytorch.org/docs/stable/notes/randomness.html#reproducibility
        
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
    
    def _setup_data(self):
        """Loads training and validation set. Creates DataLoader for training."""
        self.train_set = get_dataset(config=self.config, split='train', seed=self.seed, path=self.data_dir)
        self.val_set = get_dataset(config=self.config, split='val', seed=self.run_seed, path=self.data_dir)
        
        self.config['_input_dim'], self.config['_output_dim'] = infer_shapes(self.train_set)
        
        # construct train loader
        _gen = torch.Generator()
        _gen.manual_seed(self.run_seed)
        self.train_loader = DataLoader(self.train_set, drop_last=True, shuffle=True, generator=_gen,
                                       batch_size=self.config['batch_size'])
        
        return

    def _setup_model(self):
        """Initializes the model."""
        torch.manual_seed(self.seed) # Reseed to have same initialization   
        torch.cuda.manual_seed_all(self.seed)        
        
        self.model = get_model(self.config)
        self.model.to(self.device)

        return
        

    def setup(self):
        
        #============ Data =================
        self._setup_data()
               
        #============ Model ================
        self._setup_model()
        print(self.model)

        #============ Loss function ========
        self.training_loss = Loss(name=self.config['loss_func'], backwards=True)
        
        #============ Optimizer ============
        opt_obj, hyperp = get_optimizer(self.config['opt'])
        
        self._init_opt(opt_obj, hyperp)
        
        self.sched = get_scheduler(self.config['opt'], self.opt)
        
        #============ Results ==============
        opt_val = self._compute_opt_value()
        
        self.results = {'config': self.config,
                        'history': {},
                        'summary': {}}
        
        if opt_val is not None:
            self.results['summary']['opt_val'] = opt_val
        
        return 
    
    def _init_opt(self, opt_obj, hyperp):
        """Initializes the opt object. If your optimizer needs custom commands, add them here."""
        
        self.opt = opt_obj(params=self.model.parameters(), **hyperp)         
        
        print(self.opt)        
        return
    
    def run(self):
        start_time = str(datetime.datetime.now())
        score_list = []   
        self._epochs_trained = 0
        
        for epoch in range(self.config['max_epoch']):
            
            # Record metrics
            score_dict = {'epoch': epoch}
            score_dict['learning_rate'] = self.sched.get_last_lr()[0] # must be stored before sched.step()
                           
            # Train one epoch
            s_time = time.time()
            self.train_epoch()
            e_time = time.time()
            
            # Record metrics
            score_dict['train_epoch_time'] = e_time - s_time       
            score_dict['model_norm'] = l2_norm(self.model)        
            score_dict['grad_norm'] = grad_norm(self.model)
                
            # Validation
            with torch.no_grad():
                
                metric_dict = {'loss': Loss(self.config['loss_func'], backwards=False), 
                               'score': Loss(self.config['score_func'], backwards=False)}      
                
                train_dict = self.evaluate(self.train_set, 
                                           metric_dict = metric_dict,
                                           )  
            
                val_dict = self.evaluate(self.val_set, 
                                         metric_dict = metric_dict,
                                         )                     
                       
                # Record metrics
                score_dict.update(train_dict)
                score_dict.update(val_dict)
                
                # Record metrics specific to MoMo methods 
                if self.opt.state.get('step_size_list'):
                    score_dict['step_size_list'] = [float(np.format_float_scientific(t,5)) for t in self.opt.state['step_size_list']] 
                    self.opt.state['step_size_list'] = list()
                # fstar estimator
                if self.opt.state.get('fstar'):
                    score_dict['fstar'] = self.opt.state['fstar']
                if self.opt.state.get('h'):
                    score_dict['h'] = self.opt.state['h']

                # Add score_dict to score_list
                score_list += [score_dict]
            self._epochs_trained += 1
        
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
                
        self.model.train()
        pbar = tqdm.tqdm(self.train_loader)
        
        for batch in pbar:
            self.opt.zero_grad()    
            
            # get batch and compute model output
            data, targets = batch['data'].to(device=self.device), batch['targets'].to(device=self.device)
            out = self.model(data).to(device=self.device)

            if len(out.shape) <= 1:
                warnings.warn(f"Shape of model output is {out.shape}, recommended to have shape [batch_size, ..].")
               
            closure = lambda: self.training_loss.compute(out, targets)
            
            # see optim/README.md for explanation 
            if hasattr(self.opt,"prestep"):
                ind = batch['ind'].to(device=self.device) # indices of batch members
                self.opt.prestep(out, targets, ind, self.training_loss.name)
            
            # Here the magic happens
            loss_val = self.opt.step(closure=closure) 
            pbar.set_description(f'Training - {loss_val:.3f}')

        
        print("Current learning rate", self.sched.get_last_lr()[0])
        
        # update learning rate             
        self.sched.step()       

        return
    
    def evaluate(self, dataset, metric_dict):
        """
        Evaluate model for a given dataset (train or val), and for several metrics.
        
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
                # metric takes average over batch ==> multiply with batch size
                score_dict[_met] += _met_fun.compute(out, targets).item() * data.shape[0] 
                
            pbar.set_description(f'Validating {dataset.split}')

            
        
        for _met in metric_dict.keys():
            # Get from sum to average
            score_dict[_met] = float(score_dict[_met] / len(dl.dataset))
            
            # add split in front of names
            score_dict[dataset.split + '_' + _met] = score_dict.pop(_met)
        
        return score_dict

    def save_checkpoint(self, path):
        """See https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html"""
        torch.save({'epoch': self._epochs_trained,
                    'model_state_dict': self.model.state_dict(),
                    'opt_state_dict': self.opt.state_dict(),
                    }, 
                   path + self.name + '.mt')

        return         

    def _compute_opt_value(self):
        """
        For linear model, the problem is convex and we can compute the optimal value
        """
        if self.config['model'] == 'linear':
            
            #fit_intercept = (self.model[0].bias is not None)
            
            if self.config['loss_func'] == 'squared':
                opt_val = ridge_opt_value(X=self.train_set.dataset.tensors[0].detach().numpy(),
                                          y=self.train_set.dataset.tensors[1].detach().numpy(),
                                          lmbda = self.config['opt'].get('weight_decay', 0),
                                          fit_intercept = False
                                          )
            elif self.config['loss_func'] == 'logistic':
                opt_val = logreg_opt_value(X=self.train_set.dataset.tensors[0].detach().numpy(),
                                            y=self.train_set.dataset.tensors[1].detach().numpy().astype(int).reshape(-1),
                                            lmbda = self.config['opt'].get('weight_decay', 0),
                                            fit_intercept = False
                                            )
        else:
            opt_val = None
            
        return opt_val
    
    

        



