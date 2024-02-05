import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional


def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


class SwinTransformer(nn.Module):
    def __init__(self,
                 patch_size=4,  # 初始embedding操作的patch size大小
                 in_chans=1,  # 输入通道数
                 embed_dim=96,  # 初始embedding操作后的通道数
                 depth=(4, 8, 8, 4),  # swimtransformer block的数量
                 num_heads=(3, 6, 12, 24),  # 多头注意力机制的数量
                 window_size=7,  # W-MSA和SW-MSA中分割成每个patch的大小
                 mlp_ratio=4,  # mlp模块中通道数增长的倍数
                 qkv_bias=True,
                 drop_rate=0,  # 最开始的drop率
                 attn_drop_rate=0,
                 drop_path_rate=0.1,  # 最后分类时的drop率,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False,
                 upsample_method='pixelshuffle',
                 upsample_rate=2,
                 **kwargs):
        super().__init__()

        in_feature = in_chans
        encoder_dim = embed_dim * 2 ** (len(depth) - 1)
        out_feature = in_chans
        self.upsample_method = upsample_method
        self.num_layer = len(depth)  # 层数
        self.embed_dim = embed_dim  # embeddinglayer后的通道数
        self.patch_norm = patch_norm
        self.decoder_feature = int(embed_dim * 2 ** (self.num_layer - 1))  # 最终输出的通道数
        self.mlp_ratio = mlp_ratio

        # 初始patch embedding操作
        self.shallow_extration = Shallow_Extra(patch_size=patch_size, in_c=in_feature,
                                               embed_dim=embed_dim, norm_layer=norm_layer if patch_norm else None)
        self.pos_drop = nn.Dropout(drop_rate)
        # 初始patch embedding操作 结束

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # 丢失率从0-0.1逐层递增

        # 建立swintransformer block
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layer):  # 总共有四层，循环四层
            layers = BasicLayer(dim=embed_dim * 2 ** i_layer,
                                depth=depth[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depth[:i_layer]):sum(depth[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=PatchMerging if (i_layer < self.num_layer - 1) else None,
                                # 判断是否需要patch merging
                                use_checkpoint=use_checkpoint)
            self.layers.append(layers)  # 把每一层加入到module列表中

        # 重构+融合  encoder_dim=1024
        self.upsample2 = Upsample(encoder_dim, upsample_rate, upsample_method)  # 8,8,768 -> 16,16,384
        self.fusion2 = GL_MultiplicationFusion(encoder_dim // 2)  # 384
        self.upsample1 = Upsample(encoder_dim // 2, upsample_rate, upsample_method)  # 16,16,384 -> 32,32,192
        self.fusion1 = GL_MultiplicationFusion(encoder_dim // 4)  # 192
        self.upsample0 = Upsample(encoder_dim // 4, upsample_rate, upsample_method)  # 32,32,192 -> 64,64,96
        self.fusion0 = GL_MultiplicationFusion(encoder_dim // 8)  # 96
        self.upsample_last = nn.PixelShuffle(upsample_rate * 2)  # 64,64,96 ->256,256,6
        self.conv_last = nn.Conv2d(encoder_dim // 128, out_feature, kernel_size=3, stride=1, padding=1)

        # 任务操作
        self.norm = norm_layer(self.decoder_feature)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        '''########################### 1. Encoder ##############################'''
        x, H, W = self.shallow_extration(x)  # 输入x = B,C,H,W, 输出x = B,H*W,C
        x = self.pos_drop(x)
        global_list = []
        for layer in self.layers:
            x, H, W, x_global = layer(x, H, W)
            global_list.append(x_global)
        x = self.norm(x)
        B, _, C = x.shape
        encoder_out = x.transpose(1, 2).view(B, C, H, W)
        '''########################### 2. Decoder ##############################'''
        if self.upsample_method == 'pixelshuffle':
            x_local = self.upsample2(encoder_out)
            x_global = global_list[2]
            x_out = self.fusion2(x_local, x_global)

            x_local = self.upsample1(x_out)
            x_global = global_list[1]
            x_out = self.fusion1(x_local, x_global)

            x_local = self.upsample0(x_out)
            x_global = global_list[0]
            x_out = self.fusion0(x_local, x_global)

            decoder_out = self.conv_last(self.upsample_last(x_out))
            return decoder_out


class Upsample(nn.Module):

    def __init__(self, local_dim, upsample_rate, upsample_method):
        super().__init__()
        self.upsample_method = upsample_method
        self.upsample_rate = upsample_rate
        if self.upsample_method == 'pixelshuffle':
            self.process = nn.Sequential(nn.Conv2d(local_dim, 2 * local_dim, kernel_size=3, stride=1, padding=1),
                                         nn.LeakyReLU(inplace=True))
            self.pixelshuffle = nn.PixelShuffle(self.upsample_rate)
        if self.upsample_method == 'nearest':
            self.process = nn.Sequential(nn.Conv2d(local_dim, 0.5 * local_dim, kernel_size=3, stride=1, padding=1),
                                         nn.LeakyReLU(inplace=True))

    def forward(self, x):
        if self.upsample_method == 'pixelshuffle':
            return self.pixelshuffle(self.process(x))
        if self.upsample_method == 'nearest':
            return self.process(nn.functional.interpolate(x, scale_factor=self.upsample_rate))


class GL_MultiplicationFusion(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=int(in_channel * 2),
                      out_channels=in_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,
                      out_channels=int(in_channel / 4), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(in_channel / 4)),
            nn.ReLU(inplace=True))

        self.conv3 = nn.Conv2d(in_channels=int(in_channel / 4),
                               out_channels=2, kernel_size=3, stride=1, padding=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x_local, x_global):
        x = torch.cat((x_local, x_global), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        attn = self.sigmoid(x)

        out = x_local * attn[:, 0, :, :].unsqueeze(1) + x_global * attn[:, 1, :, :].unsqueeze(1)

        return out


class Shallow_Extra(nn.Module):
    '''
    卷积提取浅层特征
    embedding操作，将图片划分成一个个patch
    输入B,C,H,W输出B,H*W,C
    '''

    def __init__(self, patch_size=4, in_c=1, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_chans = in_c
        self.embed_dim = embed_dim
        self.shallow_extra = nn.Sequential(
            nn.Conv2d(in_c, embed_dim // 2, kernel_size=patch_size, stride=patch_size),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=1),
            nn.LeakyReLU(inplace=True)
        )

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape

        # 如果输入图片的尺寸不是patchsize的整数倍，需要对其填0操作
        if (H % self.patch_size[0] != 0) or (W % self.patch_size[0] != 0):
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
                          0, self.patch_size[0] - H % self.patch_size[0],
                          0, 0))

        x = self.shallow_extra(x)  # 提取浅层特征
        _, _, H, W = x.shape  # x = B,C,H,W
        x = x.flatten(2).transpose(1, 2)  # x = B,H*W,C
        x = self.norm(x)
        return x, H, W


class PatchMerging(nn.Module):
    '''
    通过swintransformer block之后，通过patchemerging层调整尺寸和通道数
    尺寸缩小一半，通道数增加一半
    '''

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x, H, W):  # x = B,H*W,C
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = x.view(B, H, W, C)
        if (H % 2 == 1) or (W % 2 == 1):  # 如果尺寸不能被2整除，填0
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # 每隔2行/2列取数
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)  # 按照通道数连接 x=B,0.5H,0.5W,4C
        x = x.view(B, -1, 4 * C)  # x = B,0.5H*0.5W,4C
        x = self.norm(x)
        x = self.reduction(x)  # x = B,0.5H*0.5W,2C
        return x


class BasicLayer(nn.Module):
    '''
    建立swintransformer block + patch emerging层
    '''

    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4, qkv_bias=True, drop=0., attn_drop=0., drop_path=0,
                 norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()

        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = window_size // 2  # 移动为0.5M，向下取整

        # 搭建该stage中的swintransformer block, W-MSA+SW-MSA
        self.block = nn.ModuleList([
            SwinTransformer_block(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,  # 先W-MSA再SW-MSA
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)
        ])

        # patch emerging layer
        if downsample is not None:
            self.downsample = downsample(dim)
        else:
            self.downsample = None

    def creat_mask(self, x, H, W):  # 用于创建mask蒙版，用于SW-MSA
        #  保证HW能被windowsize整除
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 建立全0的空白mask
        h_slice = (slice(0, -self.window_size),
                   slice(-self.window_size, -self.shift_size),
                   slice(-self.shift_size, None))
        w_slice = (slice(0, -self.window_size),
                   slice(-self.window_size, -self.shift_size),
                   slice(-self.shift_size, None))
        cnt = 0
        for h in h_slice:
            for w in w_slice:
                img_mask[:, h, w, :] = cnt  # 切割img_mask区域，并填上不同的数字
                cnt += 1

        # 将划分好区域的mask进行分割
        mask_window = window_partition(img_mask, self.window_size)  # B*H*W/M/M, M, M, 1
        mask_window = mask_window.view(-1, self.window_size * self.window_size)  # B*H*W/M/M, M*M
        attn_mask = mask_window.unsqueeze(1) - mask_window.unsqueeze(2)  # 广播机制 B*H*W/M/M, M*M, M*M
        # attn_mask中每一行填0的代表要计算的像素区域
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, H, W):
        attn_mask = self.creat_mask(x, H, W)  # 建立蒙版
        for blk in self.block:  # 每一个循环计算得到一个stage的结果，遍历整个循环就完成了所有swintransformer模块的计算
            blk.H = H
            blk.W = W
            # x = blk(x, attn_mask)
            x = blk(x, attn_mask)
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"
        global_feature = x.view(B, H, W, -1)
        global_feature = global_feature.permute(0, 3, 1, 2)

        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = (H + 1) // 2, (W + 1) // 2
        return x, H, W, global_feature


class SwinTransformer_block(nn.Module):
    """
    建立W-MSA+SW-MSA
    """

    def __init__(self, dim,
                 num_heads,
                 window_size=7,
                 shift_size=0,
                 mlp_ratio=4,
                 qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm
                 ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)

        self.attn = WindowAttention(dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
                                    qkv_bias=qkv_bias,
                                    attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Dropout(drop_path)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, "wrong"
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        pad_l = 0
        pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # 判断使用W-MSA还是SW-MSA，使用SW-MSA需要移动特定行和列
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size)  # 将输入划分成一个个windows，划分的数量在batch上叠加 B*H*W/M/M, M, M, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # 展平成一个个tokens B*H*W/M/M, M*M, C

        # 计算W-MSA/SW-MSA的结果
        attn_windows = self.attn(x_windows, attn_mask)

        # 计算得到的结果合并成原有大小
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B,H,W,C

        # 如果使用W-MSA不需要移动回来，使用SW-MSA需要把行和列移动回来
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.window_size, self.window_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class WindowAttention(nn.Module):
    '''
    实现W-MSA和SW-MSA的注意力机制计算
    '''

    def __init__(self,
                 dim,
                 window_size,
                 num_heads,
                 qkv_bias=True,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads  # 每一个head对应的通道数
        self.scale = head_dim ** -0.5

        # 定义一个可训练的相对偏置表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads))

        # 生成相对位置偏置
        coord_h = torch.arange(self.window_size[0])
        coord_w = torch.arange(self.window_size[1])
        coord = torch.stack(torch.meshgrid([coord_h, coord_w]))  # 2, M, M
        coord_flatten = torch.flatten(coord, 1)  # 2, M*M
        relative_coord = coord_flatten[:, :, None] - coord_flatten[:, None, :]
        relative_coord = relative_coord.permute(1, 2, 0).contiguous()
        # 行标列标都加上M-1，行标再乘2M-1，最后把行和列标相乘
        relative_coord[:, :, 0] += self.window_size[0] - 1
        relative_coord[:, :, 1] += self.window_size[1] - 1
        relative_coord[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coord.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)  # 对偏置table初始化参数
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        '''
        :param x: 输入是由windowsize分割好的一个个窗口展平后的token，分割的窗口的数量叠加在batch上。  x=B*H*W/M/M, M*M, C
        :param mask: 之前建立的mask蒙版，填0的为需要计算attn的， mask = H*W/M/M, M*M, M*M
        '''
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_feature, hidden_feature=None, out_feature=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_feature = out_feature or in_feature
        hidden_feature = hidden_feature or in_feature
        self.fc1 = nn.Linear(in_feature, hidden_feature)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_feature, out_feature)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def window_partition(x, window_size: int):
    """
    将特征图按照window_size分成一个个window   x=B,H,W,C
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    # windows = B*H*W/M/M , M , M , C  也就是把一张图分的数量全部叠加到batch上，每个wimdow大小是M*M*C
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    '''
    将一个个windows重新还原成原始大小的特征图  windows = B*H*W/M/M, M, M, C
    '''
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x  # x = B,H,W,C


def swin_model():
    model = SwinTransformer(in_chans=1,
                            patch_size=4,
                            window_size=8,
                            embed_dim=128,
                            depths=(4, 8, 8, 4),
                            num_heads=(4, 8, 16, 32))
    return model
