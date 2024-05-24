import PIL
import time
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from einops import rearrange
import torch.nn.init as init


class CrossEntropy2d(nn.Module):
    def __init__(self, size_average=None, ignore_label=255, reduction: str = 'mean'):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label
        self.reduction = reduction

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return torch.zeros(1)
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, reduction=self.reduction)
        return loss


def adjust_learning_rate(optimizer, base_lr, i_iter, max_iter, power=0.9):
    lr = base_lr * ((1 - float(i_iter) / max_iter) ** (power))
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def scaled_l2(X, C, S):
    """
    scaled_l2 distance
    Args:
        X (b*n*d):  original feature input
        C (k*d):    code words, with k codes, each with d dimension
        S (k):      scale cofficient
    Return:
        D (b*n*k):  relative distance to each code
    Note:
        apparently the X^2 + C^2 - 2XC computation is 2x faster than
        elementwise sum, perhaps due to friendly cache in gpu
    """
    assert X.shape[-1] == C.shape[-1], "input, codeword feature dim mismatch"
    assert S.numel() == C.shape[0], "scale, codeword num mismatch"

    b, n, d = X.shape
    X = X.view(-1, d)  # [bn, d]
    Ct = C.t()  # [d, k]
    X2 = X.pow(2.0).sum(-1, keepdim=True)  # [bn, 1]
    C2 = Ct.pow(2.0).sum(0, keepdim=True)  # [1, k]
    norm = X2 + C2 - 2.0 * X.mm(Ct)  # [bn, k]
    scaled_norm = S * norm
    D = scaled_norm.view(b, n, -1)  # [b, n, k]
    return D


def aggregate(A, X, C):
    """
    aggregate residuals from N samples
    Args:
        A (b*n*k):  weight of each feature contribute to code residual
        X (b*n*d):  original feature input
        C (k*d):    code words, with k codes, each with d dimension
    Return:
        E (b*k*d):  residuals to each code
    """
    assert X.shape[-1] == C.shape[-1], "input, codeword feature dim mismatch"
    assert A.shape[:2] == X.shape[:2], "weight, input dim mismatch"
    X = X.unsqueeze(2)  # [b, n, d] -> [b, n, 1, d]
    C = C[None, None, ...]  # [k, d] -> [1, 1, k, d]
    A = A.unsqueeze(-1)  # [b, n, k] -> [b, n, k, 1]
    R = (X - C) * A  # [b, n, k, d]
    E = R.sum(dim=1)  # [b, k, d]
    return E


def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class Residual2(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, x2, **kwargs):
        return self.fn(x, x2, **kwargs) + x


# 等于 PreNorm
class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNorm2(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, x2, **kwargs):
        return self.fn(self.norm(x), self.norm(x2), **kwargs)


class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Cross_Attention(nn.Module):
    def __init__(self, dim, heads=1, dropout=0., softmax=True):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.softmax = softmax
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.nn1 = nn.Linear(dim, dim)
        self.do1 = nn.Dropout(dropout)
        # self.to_out = nn.Sequential(
        #     nn.Linear(dim, dim),
        #     nn.Dropout(dropout)
        # )

    def forward(self, x, m, mask=None):

        b, n, _, h = *x.shape, self.heads
        q = self.to_q(x)
        k = self.to_k(m)
        v = self.to_v(m)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), [q, k, v])

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        if self.softmax:
            attn = dots.softmax(dim=-1)
        else:
            attn = dots
        # attn = dots
        # vis_tmp(dots)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.nn1(out)
        out = self.do1(out)

        return out


class Attention(nn.Module):

    def __init__(self, dim, heads=1, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)  # Wq,Wk,Wv for each vector, thats why *3
        # torch.nn.init.xavier_uniform_(self.to_qkv.weight)
        # torch.nn.init.zeros_(self.to_qkv.bias)

        self.nn1 = nn.Linear(dim, dim)
        # torch.nn.init.xavier_uniform_(self.nn1.weight)
        # torch.nn.init.zeros_(self.nn1.bias)
        self.do1 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # split into multi head attentions

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)  # follow the softmax,q,d,v equation in the paper

        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # product of v times whatever inside softmax
        out = rearrange(out, 'b h n d -> b n (h d)')  # concat heads into one matrix, ready for next encoder block
        out = self.nn1(out)
        out = self.do1(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attention, mlp in self.layers:
            x = attention(x, mask=mask)  # go to attention
            x = mlp(x)  # go to MLP_Block
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout, softmax=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual2(PreNorm2(dim, Cross_Attention(dim, heads=heads,
                                                        dropout=dropout,
                                                        softmax=softmax))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, m, mask=None):
        """target(query), memory"""
        for attn, ff in self.layers:
            x = attn(x, m, mask=mask)
            x = ff(x)
        return x


class WFSS(nn.Module):
    def __init__(self, num_features=103, num_classes=9, conv_features=64, trans_features=32, K=48, D=32,
                 emb_dropout=0.1, dim=64, num_tokens=64, depth=1, heads=1, mlp_dim=64, dropout=0.1):
        super(WFSS, self).__init__()

        self.conv0 = nn.Conv2d(num_features, conv_features, kernel_size=3, stride=1, padding=0, dilation=1,
                               bias=True)
        self.conv1 = nn.Conv2d(conv_features, conv_features, kernel_size=3, stride=1, padding=0, dilation=2,
                               bias=True)
        self.conv2 = nn.Conv2d(conv_features, conv_features, kernel_size=3, stride=1, padding=0, dilation=3,  # 3
                               bias=True)

        self.alpha3 = nn.Conv2d(conv_features, trans_features, kernel_size=1, stride=1, padding=0,
                                bias=False)
        self.beta3 = nn.Conv2d(conv_features, trans_features, kernel_size=1, stride=1, padding=0,
                               bias=False)
        self.gamma3 = nn.Conv2d(conv_features, trans_features, kernel_size=1, stride=1, padding=0,
                                bias=False)
        self.deta3 = nn.Conv2d(trans_features, conv_features, kernel_size=1, stride=1, padding=0,
                               bias=False)

        self.encoding = nn.Conv2d(conv_features, D, kernel_size=1, stride=1, padding=0,
                                  bias=False)

        self.codewords = nn.Parameter(torch.Tensor(K, D), requires_grad=True)
        self.scale = nn.Parameter(torch.Tensor(K), requires_grad=True)
        self.attention = nn.Linear(D, conv_features)

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.conv_cls = nn.Conv2d(conv_features * 3, num_classes, kernel_size=1, stride=1, padding=0,
                                  bias=True)

        self.drop = nn.Dropout(0.5)
        self.conv_features = conv_features
        self.trans_features = trans_features
        self.K = K
        self.D = D

        std1 = 1. / ((self.K * self.D) ** (1 / 2))
        self.codewords.data.uniform_(-std1, std1)
        self.scale.data.uniform_(-1, 0)
        self.BN = nn.BatchNorm1d(K)

        # transformer---------------------------------------------------------------------------------------
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.empty(1, (num_tokens + 1), dim))
        torch.nn.init.normal_(self.pos_embedding, std=.02)
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.dropout = nn.Dropout(emb_dropout)

        self.to_cls_token = nn.Identity()

        self.transformer_decoder = TransformerDecoder(dim, depth, heads=1, mlp_dim=mlp_dim, dropout=0)
        # self.pos_embedding_decoder =nn.Parameter(torch.randn(1, conv_features, 148, 80))

    def _forward_transformer_decoder(self, x, m, pos_embedding_decoder):
        b, c, h, w = x.shape
        x = x + pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b c (h w)')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b c (h w) -> b c h w', h=h)
        return x

    def forward(self, x, mask=None):
        interpolation = nn.UpsamplingBilinear2d(size=x.shape[2:4])
        x = self.relu(self.conv0(x))  # C1
        conv1 = x

        x = self.relu(self.conv1(x))  # C2
        conv2 = x
        x = self.avgpool(x)  # P1

        x = self.relu(self.conv2(x))
        conv3 = x  # C3
        # n, c, h, w = x.size()
        interpolation_context3 = nn.UpsamplingBilinear2d(size=x.shape[2:4])

        x_half = self.avgpool(x)  # P2
        b, c, h, w = x_half.size()

        # transformer 光谱信息------------------------------------------------------------#
        T = rearrange(x_half, 'b c h w -> b c (h w)')
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        T = torch.cat((cls_tokens, T), dim=1)
        T += self.pos_embedding
        x = self.dropout(T)
        x = self.transformer(x, mask)

        # decoder
        pos_embedding_decoder = nn.Parameter(torch.randn(1, 64, h, w)).cuda()
        x_trans = self._forward_transformer_decoder(x_half, x, pos_embedding_decoder)
        x_trans = interpolation_context3(x_trans)

        # self-attention 空间信息--------------------------------------------------------#
        alpha_x = self.alpha3(x_half)
        beta_x = self.beta3(x_half)
        gamma_x = self.relu(self.gamma3(x_half))

        alpha_x = alpha_x.squeeze().permute(1, 2, 0)  # 从n,c,h,w变成h,w,c
        # h*w x c
        alpha_x = alpha_x.view(-1, self.trans_features)  # 从h,w,c变成h*w,c    对应论文中(hw/16)*n
        # c x h*w
        beta_x = beta_x.view(self.trans_features, -1)  # 对应论文中n*(hw/16)
        gamma_x = gamma_x.view(self.trans_features, -1)

        context_x = torch.matmul(alpha_x, beta_x)
        context_x = F.softmax(context_x, dim=1)

        context_x = torch.matmul(gamma_x, context_x)
        context_x = context_x.view(b, self.trans_features, h, w)  # 对应论文(h/4)*(w/4)*n
        context_x = interpolation_context3(context_x)  # 对应论文(h/2)*(w/2)*n

        deta_x = self.relu(self.deta3(context_x))  # F(U(B))

        x = deta_x + conv3

        Z = self.relu(self.encoding(x)).view(1, self.D, -1).permute(0, 2, 1)  # n,h*w,D

        A = F.softmax(scaled_l2(Z, self.codewords, self.scale), dim=2)  # b,n,k
        E = aggregate(A, Z, self.codewords)  # b,k,d
        E_sum = torch.sum(self.relu(self.BN(E)), 1)  # b,d
        gamma = self.sigmoid(self.attention(E_sum))  # b,num_conv
        gamma = gamma.view(-1, self.conv_features, 1, 1)
        x = x + x * gamma

        # x_trans = interpolation(x_trans)
        f = 0.1
        x = f*x + (1-f)*x_trans
        context3 = interpolation(x)
        conv2 = interpolation(conv2)
        conv1 = interpolation(conv1)

        x = torch.cat((conv1, conv2, context3), 1)
        x = self.conv_cls(x)

        return x