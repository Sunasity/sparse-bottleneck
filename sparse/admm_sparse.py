import torch
import torch.nn as nn
from sparse.sparse import SparseParam
import copy

class ADMMParameter(SparseParam):
    def __init__(self, conf):
        super(ADMMParameter, self).__init__(conf)
        if conf['scheme'] != 'admm':
            raise Exception('scheme must be set as admm when using ADMM pruning.')
        self.rho = float(conf['rho'])
        self.update_step = int(conf['update_step'])
        self.Zs = []
        self.Us = []

    def InitParameter(self, model):
        _idx = 0
        _pruned_idx = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                if _idx in self.pruned_idx:
                    with torch.no_grad():
                        self.Us.append(torch.zeros_like(m.weight))
                        self.Zs.append(self._PruneWeight(copy.deepcopy(m.weight), self.expected_ratio[_pruned_idx]))
                    _pruned_idx += 1
                _idx += 1

    def ComputeRegulier(self, regulier, model):
        _idx = 0
        _pruned_idx = 0
        _loss = 0.0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                if _idx in self.pruned_idx:
                    _loss += torch.sqrt(regulier(m.weight + self.Us[_pruned_idx], self.Zs[_pruned_idx]))
                    _pruned_idx += 1
                _idx += 1
        return self.rho * _loss

    def UpdateParameter(self, model):
        _idx = 0
        _pruned_idx = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                if _idx in self.pruned_idx:
                    with torch.no_grad():
                        self.Zs[_pruned_idx] = self._PruneWeight(m.weight + self.Us[_pruned_idx], self.expected_ratio[_pruned_idx])
                        self.Us[_pruned_idx].add_(m.weight - self.Zs[_pruned_idx])
                    _pruned_idx += 1
                _idx += 1

    def CheckDistribution(self, model):
        _idx = 0
        _pruned_idx = 0
        means = []
        vars = []
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                if _idx in self.pruned_idx:
                    with torch.no_grad():
                        means.append(torch.mean(m.weight))
                        vars.append(torch.var(m.weight))
                    _pruned_idx += 1
                _idx += 1
        return zip(means, vars)

    def _PruneWeight(self, weight, ratio):
        flatten_weight = torch.flatten(torch.abs(weight))
        sorted, _ = torch.sort(flatten_weight)
        index = int(ratio*flatten_weight.size()[0])
        threshold = sorted[index]
        weight[torch.abs(weight)<threshold] = 0.0
        return weight



