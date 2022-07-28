import torch
from torch import nn


class CombineImpl():
    def combine(self, state_dicts, final_size_ratio=1):
        now_conv_modules = []
        now_linear_modules = []
        for name, module in self._actor.named_modules():
            if isinstance(module, nn.Conv2d):
                now_conv_module.append((name, module))
            elif isinstance(module, nn.Linear):
                now_linear_module.append((name, module))
        tasks_num = len(state_dicts)
        conv_modules_list = []
        linear_modules_list = []
        if tasks_num > 0:
            for name, conv_module in now_conv_modules:
                conv_modules_list.append((name, [conv_module] + [state_dict[name + '.weight'] for state_dict in state_dicts]))
                try:
                    conv_modules_list.append((name, [conv_module] + [state_dict[name + '.bias'] for state_dict in state_dicts]))
                except:
                    pass
            for name, linear_module in now_linear_modules:
                linear_modules_list.append([linear_module] + [state_dict[name + '.weight'] for state_dict in state_dicts])
                try:
                    linear_modules_list.append([linear_module] + [state_dict[name + '.bias'] for state_dict in state_dicts])
                except:
                    pass
        for name, conv_modules in conv_modules_list:
            if self._combine_method == 'addon':
                combine_module = torch.mean(conv_modules)
                combine_state_dict
