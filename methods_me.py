import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from optimizers import get_optimizer
from backbones import networks
from FDA_1d import FDA_1d_with_fs



class Algorithm(torch.nn.Module):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def forward(self, x):
        return self.predict(x)

    def new_optimizer(self, parameters):
        # if name is None:
        #     name=self.hparams["optimizer"]
        optimizer = get_optimizer(self.hparams,parameters)
            
        return optimizer

    def clone(self):
        clone = copy.deepcopy(self)
        clone.optimizer = self.new_optimizer(clone.network.parameters())
        clone.optimizer.load_state_dict(self.optimizer.state_dict())

        return clone

class Baseline(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Baseline, self).__init__(input_shape, num_classes, num_domains, hparams)
        
        print("Building F")
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        # gesture network
        self.classifier_g = nn.Linear(self.featurizer.n_outputs, num_classes)
        # self.classifier_g = networks.MLP(self.featurizer.n_outputs, num_classes,self.hparams)
        # style network
        # self.classifier_s = nn.Linear(self.featurizer.n_outputs, num_domains)
        # self.classifier_s = networks.MLP(self.featurizer.n_outputs, num_domains, self.hparams)
        
        
        self.optimizer_f = self.new_optimizer(self.featurizer.parameters())
        self.optimizer_g = self.new_optimizer(self.classifier_g.parameters())
        # self.optimizer_s = self.new_optimizer(self.classifier_s.parameters())
        # self.weight_adv = hparams["w_adv"]

    def forward_g(self, x):
        # learning gesture network on randomized style
        # return self.classifier_g(self.randomize(self.featurizer(x), "style"))
        return self.classifier_g(self.featurizer(x))


    def update(self, minibatches,all_x_test,unlabeled=None):
        # all_x = torch.cat([x for x, y in minibatches])
        all_x=minibatches[0]
        # all_y = torch.cat([y for x, y in minibatches])
        all_y=minibatches[1]
        

        all_x_src_to_tar=FDA_1d_with_fs(all_x,all_x_test)
        # all_d = torch.cat([
        #     torch.full((x.shape[0],), i, dtype=torch.int64, device="cuda")
        #     for i, (x, y) in enumerate(minibatches)
        # ])

        # learn gesture feature
        self.optimizer_f.zero_grad()
        self.optimizer_g.zero_grad()
        loss_g = F.cross_entropy(self.forward_g(all_x_src_to_tar), all_y)
        loss_g.backward()
        self.optimizer_f.step()
        self.optimizer_g.step()  

        return {
            "loss_g": loss_g.item(),
        }

    def predict(self, x):
        return self.classifier_g(self.featurizer(x))  