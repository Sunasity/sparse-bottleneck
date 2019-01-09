import torch
import torch.nn as nn

class SparseParam(object):
    def __init__(self, conf):
        self.scheme = conf['scheme']
        self.sparse_part = conf['sparse_part']
        self.sparse_option = {'fc': nn.Linear, 'conv': nn.Conv2d}
        # higher ratio, higher sparsity. 
        self.expected_ratio = [float(x) for x in conf['expected_ratio'].split(',')]
        # the idx of layers to be pruned. 
        self.pruned_idx = [int(x) for x in conf['pruned_idx'].split(',')]
        # whether all conv layers in the block uses the same ratio. only active for bottleneck like models
        # such as ResNet
        self.apply_bolck = bool(conf['apply_block'])
        self.block_dict = {}
    def GenBlockDict(self, model):
        pass



class PruningWeight(object):
    def __init__(self, sparse_param=None):
        self.MaskList = []
        self.threshold = []
        self.sparse_param = sparse_param

    def Init(self, model):
        _idx = 0
        if not self.sparse_param.apply_bolck:
            for m in model.modules():            
                if isinstance(m, self.sparse_param.sparse_option[self.sparse_param.sparse_part]):                
                    if _idx in self.sparse_param.pruned_idx:
                        with torch.no_grad():
                            m.weight.copy_(self._SetUpPruning(m.weight))
                        _idx += 1
        else:
            self.sparse_param.GenBlockDict(model)
            for _idx_ in self.sparse_param.pruned_idx:
                for _blocks in self.sparse_param.block_dict[_idx_]:
                    for m in _blocks.modules():
                        if isinstance(m, nn.Conv2d):
                            with torch.no_grad():
                                m.weight.copy_(self._SetUpPruning(m.weight))

    def _SetUpPruning(self, weight):
        _threshold = self._FindMidValue(weight, self.sparse_param.expected_ratio)
        sparse_weight, _Mask = self._InitMask(weight, _threshold)
        self.threshold.append(_threshold)
        self.MaskList.append(_Mask)
        return sparse_weight

    def _FindMidValue(self, weight, ratio):
        flatten_weight = torch.flatten(torch.abs(weight))
        sorted, _ = torch.sort(flatten_weight)
        index = int(ratio*flatten_weight.size()[0])
        threshold = sorted[index]
        return threshold

    def _InitMask(self, w, threshold):
        mask = torch.abs(w).ge(threshold).type(dtype=torch.float32)
        w[torch.abs(w)<threshold] = 0.0
        return w, mask

    def RecoverSparse(self, model):
        _idx = 0
        if not self.sparse_param.apply_bolck:
            for m in model.modules():
                if isinstance(m, self.sparse_param.sparse_option[self.sparse_param.sparse_part]):
                    if _idx in self.sparse_param.pruned_idx:
                        with torch.no_grad():
                            m.weight.copy_(m.weight * self.MaskList[_idx])
                        _idx += 1
        else:
            for _idx_ in self.sparse_param.pruned_idx:
                for _blocks in self.sparse_param.block_dict[_idx_]:
                    for m in _blocks.modules():
                        if isinstance(m, nn.Conv2d):
                            with torch.no_grad():
                                m.weight.copy_(m.weight * self.MaskList[_idx])
                            _idx += 1
