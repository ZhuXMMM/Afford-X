import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.ops import DeformConv2d

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.size(0)
        query = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores += attn_mask
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        return self.out_proj(context)

class CrossModalMultiHeadAttention(nn.Module):
    def __init__(self, v_dim, l_dim, num_heads, dropout=0.1, drop_path=0.1, init_values=1e-4):
        super().__init__()
        self.layer_norm_v = nn.LayerNorm(v_dim)
        self.layer_norm_l = nn.LayerNorm(l_dim)

        self.attn_v_to_l = MultiHeadAttention(embed_dim=v_dim, num_heads=num_heads)
        self.attn_l_to_v = MultiHeadAttention(embed_dim=l_dim, num_heads=num_heads)

        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.gamma_v = nn.Parameter(torch.full((v_dim,), init_values, requires_grad=True))
        self.gamma_l = nn.Parameter(torch.full((l_dim,), init_values, requires_grad=True))

    def forward(self, v, l):
        v_norm = self.layer_norm_v(v)
        l_norm = self.layer_norm_l(l)

        updated_v = self.attn_v_to_l(v_norm, l_norm, l_norm)
        updated_l = self.attn_l_to_v(l_norm, v_norm, v_norm)

        v = v + self.drop_path(self.gamma_v * self.dropout(updated_v))
        l = l + self.drop_path(self.gamma_l * self.dropout(updated_l))

        return v, l

class VisionLanguageFusionModel(nn.Module):
    def __init__(self, v_dim, l_dim, num_heads, fusion_layers=1, dropout=0.1, drop_path=0.1, use_dyhead=False, use_txtlayer=False):
        super().__init__()
        self.fusion_layers = nn.ModuleList([
            CrossModalMultiHeadAttention(v_dim, l_dim, num_heads, dropout, drop_path) for _ in range(fusion_layers)
        ])
        self.dyhead_modules = nn.ModuleList([DyHeadModule(v_dim) for _ in range(fusion_layers)])
        self.roberta_layers = nn.ModuleList([RobertaLayer(l_dim) for _ in range(fusion_layers)])
        self.use_dyhead = use_dyhead
        self.use_txtlayer = use_txtlayer

    def forward(self, image_features, text_features):
        N, C, H, W = image_features.shape

        for i in range(len(self.fusion_layers)):
            image_features_processed, text_features_processed = self.fusion_layers[i](
                image_features.view(N, C, H*W).transpose(1, 2), text_features.permute(1, 0, 2))

            image_features_temp = image_features.view(N, C, H*W).transpose(1, 2) + image_features_processed
            image_features = image_features_temp.transpose(1, 2).view(N, C, H, W)
            text_features_temp = text_features.permute(1, 0, 2) + text_features_processed
            text_features = text_features_temp.permute(1, 0, 2)

            if self.use_dyhead:
                image_features = self.dyhead_modules[i](image_features)
            if self.use_txtlayer:
                text_features = self.roberta_layers[i](text_features)

        return image_features, text_features

class RobertaLayer(nn.Module):
    def __init__(self, d_model):
        super(RobertaLayer, self).__init__()
        self.d_model = d_model
        self.attention = nn.MultiheadAttention(d_model, 8, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

        self.init_weights()

    def forward(self, x):
        residual = x
        x, _ = self.attention(x, x, x)
        x = self.dropout(x)
        x = residual + x
        x = self.norm1(x)

        residual = x
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = residual + x
        x = self.norm2(x)

        return x

    def init_weights(self):
        nn.init.kaiming_normal_(self.linear1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.linear2.weight, mode='fan_in', nonlinearity='relu')

        nn.init.constant_(self.norm1.weight, 1.0)
        nn.init.constant_(self.norm1.bias, 0)
        nn.init.constant_(self.norm2.weight, 1.0)
        nn.init.constant_(self.norm2.bias, 0)

        nn.init.xavier_uniform_(self.attention.in_proj_weight)
        nn.init.constant_(self.attention.in_proj_bias, 0)
        nn.init.xavier_uniform_(self.attention.out_proj.weight)
        nn.init.constant_(self.attention.out_proj.bias, 0)

class DyHeadModule(nn.Module):
    def __init__(self, d_model, use_dynamic=False, use_deform=False):
        super(DyHeadModule, self).__init__()

        self.use_dynamic = use_dynamic
        self.use_deform = use_deform
        self.conv_layers = nn.ModuleList()

        for _ in range(3):
            if self.use_deform:
                conv = DeformConv2d(d_model, d_model, kernel_size=3, padding=1)
            else:
                conv = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)
            self.conv_layers.append(conv)

        if self.use_dynamic:
            self.attn_conv = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(d_model, 1, kernel_size=1),
                nn.ReLU(inplace=True),
                h_sigmoid()
            )

        if self.use_deform:
            self.offset = nn.Conv2d(d_model, 27, kernel_size=3, padding=1)

        self.init_weights()

    def init_weights(self):
        for conv in self.conv_layers:
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 0)
        if self.use_dynamic:
            if self.attn_conv is not None:
                for m in self.attn_conv.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.normal_(m.weight.data, 0, 0.01)
                        if m.bias is not None:
                            m.bias.data.zero_()

    def forward(self, x):
        for conv in self.conv_layers:
            conv_args = dict()
            if self.use_deform:
                offset_mask = self.offset(x)
                offset = offset_mask[:, :18, :, :]
                mask = offset_mask[:, 18:, :, :].sigmoid()
                conv_args = dict(offset=offset, mask=mask)
            
            x = conv(x, **conv_args) if self.use_deform else conv(x)

            if self.use_dynamic:
                attn = self.attn_conv(x)
                x = x * attn

        return F.relu(x)

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True, h_max=1):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
        self.h_max = h_max

    def forward(self, x):
        return self.relu(x + 3) * self.h_max / 6
