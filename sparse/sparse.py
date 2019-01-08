import torch
import torch.nn as nn

class SparseParam(object):
    def __init__(self):
        self.scheme = 'naive'
        self.sparse_part = 'pruning_fc'
        self.sparse_option = {'pruning_fc': nn.Linear, 'pruning_conv': nn.Conv2d}
        # higher ratio, higher sparsity
        self.expected_ratio = 0.9
        self.pruned_idx = [0]
        # whether all conv layers in the block uses the same ratio. only active for bottleneck like models
        # such as ResNet
        self.apply_bolck = False
        self.block_dict = {}
    def GenBlockDict(self, model):
        self.block_dict = {1: model.module.layer1, 2: model.module.layer2, 3: model.module.layer3, 4: model.module.layer4}

sparse_param = SparseParam()

class PruningWeight(object):
    def __init__(self):
        self.MaskList = []
        self.threshold = []

    def Init(self, model):
        _idx = 0
        if not sparse_param.apply_bolck:
            for m in model.modules():            
                if isinstance(m, sparse_param.sparse_option[sparse_param.sparse_part]):                
                    if _idx in sparse_param.pruned_idx:
                        with torch.no_grad():
                            m.weight.copy_(self._SetUpPruning(m.weight))
                        _idx += 1
        else:
            sparse_param.GenBlockDict(model)
            for _idx_ in sparse_param.pruned_idx:
                for _blocks in sparse_param.block_dict[_idx_]:
                    for m in _blocks.modules():
                        if isinstance(m, nn.Conv2d):
                            with torch.no_grad():
                                m.weight.copy_(self._SetUpPruning(m.weight))

    def _SetUpPruning(self, weight):
        _threshold = self._FindMidValue(weight, sparse_param.expected_ratio)
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
        if not sparse_param.apply_bolck:
            for m in model.modules():
                if isinstance(m, sparse_param.sparse_option[sparse_param.sparse_part]):
                    if _idx in sparse_param.pruned_idx:
                        with torch.no_grad():
                            m.weight.copy_(m.weight * self.MaskList[_idx])
                        _idx += 1
        else:
            for _idx_ in sparse_param.pruned_idx:
                for _blocks in sparse_param.block_dict[_idx_]:
                    for m in _blocks.modules():
                        if isinstance(m, nn.Conv2d):
                            with torch.no_grad():
                                m.weight.copy_(m.weight * self.MaskList[_idx])
                            _idx += 1
