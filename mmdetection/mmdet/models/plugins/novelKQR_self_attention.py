import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import kaiming_init
from mmdet.ops import DeformConv

class NovelKQRAttention(nn.Module):
    """Modified GeneralizedAttention module.

    See 'An Empirical Study of Spatial Attention Mechanisms in Deep Networks'
    (https://arxiv.org/abs/1711.07971) for details.

    Args:
        in_dim (int): Channels of the input feature map.
        spatial_range (int): The spatial range.
            -1 indicates no spatial range constraint.
        num_heads (int): The head number of empirical_attention module.
        position_embedding_dim (int): The position embedding dimension.
        position_magnitude (int): A multiplier acting on coord difference.
        kv_stride (int): The feature stride acting on key/value feature map.
        q_stride (int): The feature stride acting on query feature map.
        attention_type (str): A binary indicator string for indicating which
            items in generalized empirical_attention module are used.
            '1000' indicates 'query and key content' (appr - appr) item,
            '0100' indicates 'query content and relative position'
              (appr - position) item,
            '0010' indicates 'key content only' (bias - appr) item,
            '0001' indicates 'relative position only' (bias - position) item.
    """

    def __init__(self,
                 in_dim,
                 spatial_range=-1,
                 num_heads=9,
                 position_embedding_dim=-1,
                 position_magnitude=1,
                 kv_stride=2,
                 q_stride=1,
                 attention_type='0110',
                 deformable_group=1,
                 dconv_stride = 1,
                 dconv_groups =1,
                 dconv_learnable_vector = True):

        super(NovelKQRAttention, self).__init__()

        # hard range means local range for non-local operation
        self.position_embedding_dim = (
            position_embedding_dim if position_embedding_dim > 0 else in_dim)

        self.position_magnitude = position_magnitude
        self.num_heads = num_heads
        self.channel_in = in_dim
        self.spatial_range = spatial_range
        self.kv_stride = kv_stride
        self.q_stride = q_stride
        self.attention_type = [bool(int(_)) for _ in attention_type]
        self.qk_embed_dim = in_dim // num_heads
        self.deformable_group = deformable_group
        self.dconv_stride = dconv_stride
        self.dconv_learnable_vector = dconv_learnable_vector
        self.dconv_groups = dconv_groups
        out_c = self.qk_embed_dim * num_heads


        if self.attention_type[2]:
            self.key_conv = nn.Conv2d(
                in_channels=in_dim,
                out_channels=out_c,
                kernel_size=1,
                bias=False)
            self.key_conv.kaiming_init = True

        if self.attention_type[1]:
            self.query_conv_offset = nn.Conv2d(
                in_channels=in_dim,
                out_channels=self.deformable_group * 18,
                kernel_size=3,
                padding=1,
                stride = self.dconv_stride,
                dilation=1,
                bias=False)

            self.query_dconv=DeformConv(
                in_channels =in_dim,
                out_channels = out_c,
                kernel_size=3,
                stride = self.dconv_stride,
                padding = 1,
                dilation =1,
                deformable_groups = self.deformable_group,
                groups= self.dconv_groups,
                bias = False)

        self.v_dim = in_dim // num_heads
        self.value_conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=self.v_dim * num_heads,
            kernel_size=1,
            bias=False)
        self.value_conv.kaiming_init = True


        if self.attention_type[2]:
            stdv = 1.0 / math.sqrt(self.qk_embed_dim * 2)
            appr_bias_value = -2 * stdv * torch.rand(out_c) + stdv
            self.appr_bias = nn.Parameter(appr_bias_value)

        if self.attention_type[1] and self.dconv_learnable_vector == True :
            stdv = 1.0 / math.sqrt(self.qk_embed_dim * 2)
            appr_bias_qRelPos = -2 * stdv * torch.rand(out_c) + stdv
            self.appr_bias_qRelPos = nn.Parameter(appr_bias_qRelPos)


        self.proj_conv = nn.Conv2d(
            in_channels=self.v_dim * num_heads,
            out_channels=in_dim,
            kernel_size=1,
            bias=True)
        self.proj_conv.kaiming_init = True
        self.gamma = nn.Parameter(torch.zeros(1))

        if self.spatial_range >= 0:
            # only works when non local is after 3*3 conv
            if in_dim == 256:
                max_len = 84
            elif in_dim == 512:
                max_len = 42

            max_len_kv = int((max_len - 1.0) / self.kv_stride + 1)
            local_constraint_map = np.ones(
                (max_len, max_len, max_len_kv, max_len_kv), dtype=np.int)
            for iy in range(max_len):
                for ix in range(max_len):
                    local_constraint_map[iy, ix,
                                         max((iy - self.spatial_range) //
                                             self.kv_stride, 0):min(
                                                 (iy + self.spatial_range +
                                                  1) // self.kv_stride +
                                                 1, max_len),
                                         max((ix - self.spatial_range) //
                                             self.kv_stride, 0):min(
                                                 (ix + self.spatial_range +
                                                  1) // self.kv_stride +
                                                 1, max_len)] = 0

            self.local_constraint_map = nn.Parameter(
                torch.from_numpy(local_constraint_map).byte(),
                requires_grad=False)

        if self.q_stride > 1:
            self.q_downsample = nn.AvgPool2d(
                kernel_size=1, stride=self.q_stride)
        else:
            self.q_downsample = None

        if self.kv_stride > 1:
            self.kv_downsample = nn.AvgPool2d(
                kernel_size=1, stride=self.kv_stride)
        else:
            self.kv_downsample = None

        self.init_weights()


    def forward(self, x_input):
        num_heads = self.num_heads

        # use empirical_attention
        if self.q_downsample is not None:
            x_q = self.q_downsample(x_input)
        else:
            x_q = x_input
        n, _, h, w = x_q.shape

        if self.dconv_stride > 1:
            h,w = h//self.dconv_stride, w//self.dconv_stride

        if self.kv_downsample is not None:
            x_kv = self.kv_downsample(x_input)
        else:
            x_kv = x_input
        _, _, h_kv, w_kv = x_kv.shape


        if self.attention_type[2]:
            proj_key = self.key_conv(x_kv).view(
                (n, num_heads, self.qk_embed_dim, h_kv * w_kv))

        if self.attention_type[1]:
            offset = self.query_conv_offset(x_q)
            from IPython import embed; embed()
            proj_query_relativePos =self.query_dconv(x_q,offset).view(n,num_heads,self.qk_embed_dim,h*w)


        # accelerate for saliency only
        if (np.sum(self.attention_type) == 1) and self.attention_type[2]:
            appr_bias = self.appr_bias.\
                view(1, num_heads, 1, self.qk_embed_dim).\
                repeat(n, 1, 1, 1)

            energy = torch.matmul(appr_bias, proj_key).\
                view(n, num_heads, 1, h_kv * w_kv)

            h = 1
            w = 1
        else:
            if not self.attention_type[0]:
                energy = torch.zeros(
                    n,
                    num_heads,
                    h,
                    w,
                    h_kv,
                    w_kv,
                    dtype=x_input.dtype,
                    device=x_input.device)

            # Key Content
            if self.attention_type[2]:
                appr_bias = self.appr_bias.\
                    view(1, num_heads, 1, self.qk_embed_dim).\
                    repeat(n, 1, 1, 1)

                energy += torch.matmul(appr_bias, proj_key).\
                        view(n, num_heads, 1, 1, h_kv, w_kv)

            # Query Content and Relative Position
            if self.attention_type[1]:

                if self.dconv_learnable_vector == False :
                    energy+=proj_query_relativePos.view(n,num_heads,h,w,1,1)

                else:
                    appr_bias_qRelPos = self.appr_bias_qRelPos.\
                        view(1,num_heads,1,self.qk_embed_dim).\
                        repeat(n,1,1,1)
                    energy+= torch.matmul(appr_bias_qRelPos,proj_query_relativePos).\
                            view(n,num_heads,h,w,1,1)


            energy = energy.view(n,num_heads,h*w,h_kv*w_kv)

        if self.spatial_range >= 0:
            cur_local_constraint_map = \
                self.local_constraint_map[:h, :w, :h_kv, :w_kv].\
                contiguous().\
                view(1, 1, h*w, h_kv*w_kv)

            energy = energy.masked_fill_(cur_local_constraint_map,
                                         float('-inf'))


        attention = F.softmax(energy, 3)


        proj_value = self.value_conv(x_kv)
        proj_value_reshape = proj_value.\
            view((n, num_heads, self.v_dim, h_kv * w_kv)).\
            permute(0, 1, 3, 2)

        out = torch.matmul(attention, proj_value_reshape).\
            permute(0, 1, 3, 2).\
            contiguous().\
            view(n, self.v_dim * self.num_heads, h, w)

        out = self.proj_conv(out)

        if self.dconv_stride > 1:
            out = F.interpolate(out,scale_factor=self.dconv_stride)

        if self.q_stride > 1:
            out = F.interpolate(out,scale_factor=self.q_stride)

        out = self.gamma * out + x_input

        return out

    def init_weights(self):
        for m in self.modules():
            if hasattr(m, 'kaiming_init') and m.kaiming_init:
                kaiming_init(
                    m,
                    mode='fan_in',
                    nonlinearity='leaky_relu',
                    bias=0,
                    distribution='uniform',
                    a=1)
