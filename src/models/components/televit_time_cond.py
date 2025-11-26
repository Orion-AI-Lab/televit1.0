"""
    Vision Transformers Need Registers
    https://arxiv.org/abs/2309.16588
    Taken from https://github.com/lucidrains/vit-pytorch
"""

import torch
from torch import nn

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d


class Attention(nn.Module):
    def __init__(self, dim, cond_dim=None, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

        self.has_cond = exists(cond_dim)

        self.film = None

        if self.has_cond:
            self.film = nn.Sequential(
                nn.Linear(cond_dim, dim * 2),
                nn.SiLU(),
                nn.Linear(dim * 2, dim * 2),
                Rearrange('b (r d) -> r b 1 d', r = 2)
            )

    def forward(self, x, cond=None):
        x = self.norm(x)

        # conditioning
        if exists(self.film):
            assert exists(cond)

            gamma, beta = self.film(cond)
            x = x * gamma + beta

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x, cond=None):
        for attn, ff in self.layers:
            if exists(cond):
                x = attn(x, cond) + x
            else:
                x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)
    

class Embedder(nn.Module):
    def __init__(self, input_dims, patch_sizes, embedding_dim):
        super().__init__()
        self.to_patch_embeddings = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        for i, input_dim in enumerate(input_dims):
            assert len(input_dim) == len(patch_sizes[i]), "Input dimension and patch size must have the same length"
            # do the patch emdedding according to the input dimension
            if len(input_dim) == 2:
                self.to_patch_embeddings.append(nn.Sequential(
                    Rearrange("b (h p1) (w p2) -> b (h w) (p1 p2)", p1 = patch_sizes[i][0], p2 = patch_sizes[i][1]),
                    nn.LayerNorm(patch_sizes[i][0] * patch_sizes[i][1]),
                    nn.Linear(patch_sizes[i][0] * patch_sizes[i][1], embedding_dim),
                    nn.LayerNorm(embedding_dim),
                ))
            elif len(input_dim) == 3:
                self.to_patch_embeddings.append(nn.Sequential(
                    Rearrange("b (h p1) (w p2) (z p3) -> b (h w z) (p1 p2 p3)", p1 = patch_sizes[i][0], p2 = patch_sizes[i][1], p3 = patch_sizes[i][2]),
                    nn.LayerNorm(patch_sizes[i][0] * patch_sizes[i][1] * patch_sizes[i][2] ),
                    nn.Linear(patch_sizes[i][0] * patch_sizes[i][1] * patch_sizes[i][2] , embedding_dim),
                    nn.LayerNorm(embedding_dim),
                )) 
            elif len(input_dim) == 4:
                self.to_patch_embeddings.append(nn.Sequential(
                    Rearrange("b (h p1) (w p2) (z p3) (t p4) -> b (h w z t) (p1 p2 p3 p4)", p1 = patch_sizes[i][0], p2 = patch_sizes[i][1], p3 = patch_sizes[i][2], p4 = patch_sizes[i][3]),
                    nn.LayerNorm(patch_sizes[i][0] * patch_sizes[i][1] * patch_sizes[i][2] * patch_sizes[i][3]),
                    nn.Linear(patch_sizes[i][0] * patch_sizes[i][1] * patch_sizes[i][2] * patch_sizes[i][3], embedding_dim),
                    nn.LayerNorm(embedding_dim),
                ))
            else:
                raise ValueError("Input dimension not supported")
        
    def forward(self, inputs):
        embeddings = []
        for i, input in enumerate(inputs):
            x = self.to_patch_embeddings[i](input)
            # x = self.linear_layers[i](x)
            embeddings.append(x)
        embeddings = torch.cat(embeddings, dim=1)
        return embeddings

class TokenDecoder(nn.Module):
    """
    Decoder for the Transformer gets input (tokens, dim) and outputs (num_classes, h, w)
    """
    def __init__(self, num_classes, c, t, h, w, cp, tp, hp, wp, dim, pool="mean"):
        super().__init__()
        self.num_classes = num_classes
        self.c = c
        self.cp = cp
        self.t = t
        self.tp = tp
        self.h = h
        self.hp = hp
        self.w = w
        self.wp = wp
        self.dim = dim
        self.pool = pool
        assert pool in ["mean", "linear"], "Pool must be either mean or linear"
        if pool == "linear":
            self.linear_head = nn.Linear(dim * (c//cp) * (t//tp), num_classes * hp * wp)
        elif pool == "mean":
            self.linear_head = nn.Linear(dim, num_classes * hp * wp)

    
    def forward(self, x):
        x = rearrange(x, "b (c t h w) dim-> b c t h w dim",
                        c=self.c//self.cp, t=self.t//self.tp, h=self.h//self.hp, w=self.w//self.wp, dim=self.dim)
        if self.pool == "mean":
            x = x.mean(dim=(1, 2))
            x = self.linear_head(x)
        elif self.pool == "linear":
            x = rearrange(x, "b c t h w dim -> b h w (c t dim)")
            x = self.linear_head(x)
        else:
            raise ValueError("Pool not supported")
        return rearrange(x, "b h w (nc hp wp) -> b nc (h hp) (w wp)", nc=self.num_classes, hp=self.hp, wp=self.wp)

class TeleViT(nn.Module):
    def __init__(self, *, input_dims, patch_dims, input_names, output_shape_from_input, num_classes, dim, depth, heads, mlp_dim, num_lead_times=16, cond_dim=32, num_register_tokens = 4, dim_head = 64, pool="mean"):
        super().__init__()        
        assert len(input_dims) == len(patch_dims), "List of input dimensions and patch dimensions must have the same length"
        assert len(input_dims) == len(input_names), "List of input dimensions and input names must have the same length"

        self.input_names = input_names
        self.input_dims = input_dims
        self.patch_dims = patch_dims
        self.num_classes = num_classes
        self.cond_dim = cond_dim
        assert output_shape_from_input in input_names, "Output name must be included in input names"
        self.output_shape_from_input = output_shape_from_input
        self.output_shape_idx = input_names.index(output_shape_from_input)

        self.embedder = Embedder(input_dims, patch_dims, dim)

        self.num_patches = 0
        self.num_input_patches = dict()
        for _, (input_dim, patch_dim, input_name) in enumerate(zip(input_dims, patch_dims, input_names)):
            assert len(input_dim) == len(patch_dim), f"Input dimension {input_dim} and patch dimension {patch_dim} must have the same length"
            self.num_input_patches[input_name] = 1
            for i in range(len(input_dim)):
                assert input_dim[i] % patch_dim[i] == 0, f"Input dimension {input_dim} must be divisible by the patch dimension {patch_dim}"
                self.num_input_patches[input_name] *= (input_dim[i] // patch_dim[i])
            print(f"Number of patches for input {input_dim} and patch {patch_dim} is {self.num_input_patches}")
            self.num_patches += self.num_input_patches[input_name]

        print(f"Total number of patches is {self.num_patches}")
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))

        self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, dim))


        self.lead_time_embedding = nn.Embedding(num_lead_times, cond_dim)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.pool = pool
        self.to_latent = nn.Identity()

        # calculate output shape per token
        token_output_shape = num_classes * patch_dims[self.output_shape_idx][-2] * patch_dims[self.output_shape_idx][-1]

        self.decoder = TokenDecoder(num_classes, *input_dims[self.output_shape_idx], *patch_dims[self.output_shape_idx], dim, pool=self.pool)

        self.linear_head = nn.Linear(dim, token_output_shape)

    def forward(self, inputs, cond_input=None):
        batch, device = inputs[0].shape[0], inputs[0].device

        if exists(cond_input):
            # subtract 1 from cond_input to get the index
            cond_input = cond_input - 1

        x = self.embedder(inputs)
        x += self.pos_embedding.to(device, dtype=x.dtype)

        r = repeat(self.register_tokens, 'n d -> b n d', b = batch)

        x, ps = pack([x, r], 'b * d')

        if exists(cond_input):
            cond =  self.lead_time_embedding(cond_input)

        x = self.transformer(x, cond=cond)

        x, _ = unpack(x, ps, 'b * d')

        # x = x.mean(dim = 1)

        x = x[:,:self.num_input_patches[self.output_shape_from_input],:]

        x = self.decoder(x)

        # x = self.to_latent(x)
        # x = self.linear_head(x)
        # x = rearrange(x, "b (hp wp) (nc h w)  -> b nc (h hp) (w wp)", 
        #               hp=self.input_dims[self.output_shape_idx][-2] // self.patch_dims[self.output_shape_idx][-2], 
        #               wp=self.input_dims[self.output_shape_idx][-1] // self.patch_dims[self.output_shape_idx][-1], 
        #               h=self.patch_dims[self.output_shape_idx][-2], w=self.patch_dims[self.output_shape_idx][-1], nc=self.num_classes)
        return x
