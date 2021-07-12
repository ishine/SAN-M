import oneflow.experimental as flow
import oneflow.experimental.nn as nn
import numpy as np


def mask_value(input,input_size):
    batch, row, col = input.shape
    input_size = input_size.numpy().tolist()
    
    mask = flow.ones_like(input)
    for i in range(batch):
        if input_size[i]<row:
            mask[i, input_size[i]:, :] = 0

    input = input * mask
    return input


class DFSMN_layer(nn.Module):
    def __init__(self, dim_in = 512 ,dim_out =512, hid_size = 2048,l_order = 20, r_order = 20, l_stride = 2, r_stride = 2):
        super(DFSMN_layer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.hid_size = hid_size
        self.l_order = l_order
        self.r_order = r_order
        self.l_stride = l_stride
        self.r_stride = r_stride

        self.linear1 = nn.Linear(dim_in,hid_size,bias = True)
        self.relu = nn.ReLU()
        self.linear2 =  nn.Linear(hid_size,dim_out , bias = False)
        self.fsmn = FSMN_layer(dim_out, dim_out, l_order, r_order, l_stride, r_stride)


    def forward(self,input,input_size):
        hid = self.linear1(input)
        hid = self.relu(hid)
        p = self.linear2(hid)
        pt = self.fsmn(p,input_size)
        out = pt + input
        return out


class FSMN_layer(nn.Module):
    def __init__(self,dim_in = 512,dim_out=512,l_order = 20,r_order = 20,l_stride = 2,r_stride = 2):
        super(FSMN_layer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.l_order = l_order
        self.r_order = r_order
        self.l_stride = l_stride
        self.r_stride = r_stride

        self.left_conv = nn.Conv1d(dim_in, dim_in, kernel_size=l_order, dilation=l_stride, bias=False, groups=dim_in)

        self.right_conv = nn.Conv1d(dim_in, dim_in, kernel_size=r_order, dilation=r_stride, bias=False, groups=dim_in)


    def forward(self,input,input_size):

        input = mask_value(input,input_size)
        input = input.transpose(1, 2)
        
        b,d,_ = input.size()
        
        left_pad = flow.zeros((b,d,self.l_stride * (self.l_order-1))).to(device=input.device)
        left_pad = flow.cat((left_pad,input),dim=2)
        left_attention = self.left_conv(left_pad)
        
        right_pad = flow.zeros((b,d,self.r_order * self.r_stride)).to(device=input.device)
        right_pad = flow.cat((input[:,:,self.r_stride:],right_pad),dim=2)
        right_attention = self.right_conv(right_pad)

        out = input + left_attention + right_attention
        out = out.transpose(1, 2)
        return out



class DFSMN_undirection_layer(nn.Module):
    def __init__(self, dim_in = 512 ,dim_out =512, hid_size = 2048,l_order = 20,  l_stride = 2,d_model=320):
        super(DFSMN_undirection_layer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.hid_size = hid_size
        self.l_order = l_order
        self.l_stride = l_stride

        self.linear1 = nn.Linear(dim_in,hid_size,bias = True)
        self.relu = nn.ReLU()
        self.linear2 =  nn.Linear(hid_size,dim_out , bias = False)
        self.fsmn = FSMN_undirection_layer(dim_out, dim_out, l_order, l_stride)

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self,input):
        hid = self.linear1(input)
        hid = self.relu(hid)
        p = self.linear2(hid)
        pt = self.fsmn(p)
        out = self.layer_norm(flow.add(pt,input))
        return out


class FSMN_undirection_layer(nn.Module):
    def __init__(self,dim_in = 512,dim_out=512,l_order = 20,l_stride = 2):
        super(FSMN_undirection_layer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.l_order = l_order
        self.l_stride = l_stride

        self.left_conv = nn.Conv1d(dim_in, dim_in, kernel_size=l_order, dilation=l_stride, bias=False, groups=dim_in)


    def forward(self, input):
        input = input.transpose(1, 2)

        b,d,_ = input.size()
        
        left_pad = flow.zeros((b,d,self.l_stride * (self.l_order-1))).to(device=input.device)
        left_pad = flow.cat((left_pad,input),dim=2)
        left_attention = self.left_conv(left_pad)

        out = input + left_attention
        out = out.transpose(1, 2)
        return out