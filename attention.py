import torch.nn as nn
import torch


class Attention(nn.Module):
    def __init__(self, local_shape, global_shape, method):
        super(Attention, self).__init__()
        self.__method = method

        self.__lcf_channel = local_shape[0]
        self.__lcf_H = local_shape[1]
        self.__lcf_W = local_shape[2]
        self.__glf_channel = global_shape

        self.__project = nn.Linear(self.__glf_channel, self.__lcf_channel, bias=False)
        self.__pc = nn.Linear(self.__lcf_channel, 1, bias=False)

    def forward(self, lcf, gbf, return_attention_map=False):
        gbf_pj = self.__project(gbf) if self.__glf_channel != self.__lcf_channel else gbf   # (bs, c)
        lcf_rs = lcf.reshape([-1, self.__lcf_channel, self.__lcf_H * self.__lcf_W])     # (bs, c, h*w)
        if self.__method == 'dp':
            c = torch.matmul(lcf_rs.transpose(1, 2), gbf_pj.unsqueeze(-1)).squeeze(-1)
            # (bs, h*w, c) matmul (bs, c, 1) -> (bs, h*w, 1) -> (bs, h*w)
        elif self.__method == 'pc':
            add = torch.add(lcf_rs.transpose(1, 2), gbf_pj.unsqueeze(-2))   # (bs, h*w, c)
            c = self.__pc(add).squeeze(-1)  # (bs, h*w)
        else:
            c = None
            lcf_rs = None
            print('No such method', self.__method)
            exit(0)
        a = torch.softmax(c, 1)     # (bs, h*w)
        ga = torch.matmul(lcf_rs, a.unsqueeze(-1)).squeeze(-1)  # (bs, c, h*w) matmul (bs, h*w, 1) -> (bs, c)
        if return_attention_map:
            return ga, a.reshape(-1, self.__lcf_H, self.__lcf_W)
        return ga
