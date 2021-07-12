import numpy as np

import torch
import torch.nn as torchnn
import torch.nn.functional as F

import oneflow.experimental as flow
import oneflow.experimental.nn as flownn
flow.enable_eager_execution()


a = flownn.relu()









def is_equal(torch_in,onflow_in):
    torch_in = torch_in.detach().numpy().reshape(-1)
    onflow_in = onflow_in.detach().numpy().reshape(-1)
    for i in range(len(torch_in)):
        if abs(torch_in[i]-onflow_in[i]) > 0.0001:
            return False
    return True



def class_():
    class Model_p(torchnn.Module):
        def __init__(self):
            super(Model_p,self).__init__()
            self.beding = torchnn.Embedding(5,4)

        def forward(self, input):
            output = self.beding(input)
            return output

    class Model_o(flownn.Module):
        def __init__(self):
            super(Model_o,self).__init__()
            self.beding = flownn.Embedding(5,4)
        def forward(self, input):
            output = self.beding(input)
            return output

    model_p = Model_p()
    model_o = Model_o()

    dic = model_o.state_dict()
    torch_params = {}
    for k in dic.keys():
        torch_params[k] = torch.from_numpy(dic[k].numpy())
    model_p.load_state_dict(torch_params)

    out_p = model_p(a_p)
    out_o = model_o(a_o)

    print(out_p)
    print(out_o)

    print(is_equal(out_p,out_o))
