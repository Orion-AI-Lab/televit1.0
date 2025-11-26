"""
    Vision Transformers Need Registers
    https://arxiv.org/abs/2309.16588
    Taken from https://github.com/lucidrains/vit-pytorch
"""

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_
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

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

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
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)
    

class Embedder(nn.Module):
    def __init__(self, input_dims, patch_sizes, embedding_dim, patch_emb="linear"):
        super().__init__()
        self.to_patch_embeddings = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        self.patch_emb = patch_emb
        self.norm = nn.LayerNorm(embedding_dim)
        for i, input_dim in enumerate(input_dims):
            assert len(input_dim) == len(patch_sizes[i]), "Input dimension and patch size must have the same length"
            # do the patch emdedding according to the input dimension
            if len(input_dim) == 2: # these are temporal inputs (b, c, t)
                # added 4/12/2024
                if self.patch_emb=='linear':
                    self.to_patch_embeddings.append(nn.Sequential(
                        Rearrange("b (h p1) (w p2) -> b (h w) (p1 p2)", p1 = patch_sizes[i][0], p2 = patch_sizes[i][1]),
                        nn.Linear(patch_sizes[i][0] * patch_sizes[i][1], embedding_dim),
                    ))
                else:
                    conv1d = nn.Conv1d(input_dim[0], embedding_dim*input_dim[0]//patch_sizes[i][0], groups=input_dim[0]//patch_sizes[i][0], 
                            kernel_size=patch_sizes[i][1], stride=patch_sizes[i][1], bias=True)
                    self.to_patch_embeddings.append(nn.Sequential(
                        conv1d,
                        Rearrange("b (e c) t -> b (c t) e", e=embedding_dim, c=input_dim[0]//patch_sizes[i][0], t=input_dim[1]//patch_sizes[i][1]),
                    ))
            elif len(input_dim) == 3: # these are spatial inputs (b, c, h, w)
                if self.patch_emb=='linear':
                    self.to_patch_embeddings.append(nn.Sequential(
                        Rearrange("b (h p1) (w p2) (z p3) -> b (h w z) (p1 p2 p3)", p1 = patch_sizes[i][0], p2 = patch_sizes[i][1], p3 = patch_sizes[i][2]),
                        nn.Linear(patch_sizes[i][0] * patch_sizes[i][1] * patch_sizes[i][2] , embedding_dim),
                    ))
                else:
                    conv2d = nn.Conv2d(input_dim[0], embedding_dim, kernel_size=(patch_sizes[i][1], patch_sizes[i][2]), stride=(patch_sizes[i][1], patch_sizes[i][2]), bias=True)
                    self.to_patch_embeddings.append(nn.Sequential(
                        conv2d,
                        Rearrange("b c h w -> b (h w) c"),
                    )) 
            elif len(input_dim) == 4: # these are spatio-temporal inputs (b, c, h, w, t)
                if self.patch_emb=='linear':
                    self.to_patch_embeddings.append(nn.Sequential(
                        Rearrange("b (h p1) (w p2) (z p3) (t p4) -> b (h w z t) (p1 p2 p3 p4)", p1 = patch_sizes[i][0], p2 = patch_sizes[i][1], p3 = patch_sizes[i][2], p4 = patch_sizes[i][3]),
                        nn.Linear(patch_sizes[i][0] * patch_sizes[i][1] * patch_sizes[i][2] * patch_sizes[i][3], embedding_dim),
                    ))
                else:
                    conv3d = nn.Conv3d(input_dim[0], embedding_dim, kernel_size=(patch_sizes[i][1], patch_sizes[i][2], patch_sizes[i][3]), stride=(patch_sizes[i][1], patch_sizes[i][2], patch_sizes[i][3]), bias=True)
                    self.to_patch_embeddings.append(nn.Sequential(
                        conv3d,
                        Rearrange("b c h w t -> b (h w t) c"),
                    ))


            else:
                raise ValueError("Input dimension not supported")
        
    def forward(self, inputs):
        embeddings = []
        for i, input in enumerate(inputs):
            x = self.to_patch_embeddings[i](input)
            embeddings.append(x)
        embeddings = torch.cat(embeddings, dim=1)
        embeddings = self.norm(embeddings)
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
        elif pool == "cls":
            self.linear_head = nn.Linear(dim, num_classes)

    
    def forward(self, x):
        if self.pool == "cls":
            return rearrange(self.linear_head(x), "b d -> b 1 1 d")

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
    def __init__(self, *, input_dims, patch_dims, input_names, output_shape_from_input, num_classes, dim, depth, heads, mlp_dim, num_register_tokens = 4, dim_head = 64, pool="mean", pos_emb='learnable', patch_emb="linear", cls_token=False):
        super().__init__()        
        assert len(input_dims) == len(patch_dims), "List of input dimensions and patch dimensions must have the same length"
        assert len(input_dims) == len(input_names), "List of input dimensions and input names must have the same length"

        self.input_names = input_names
        self.input_dims = input_dims
        self.patch_dims = patch_dims
        self.num_classes = num_classes
        assert output_shape_from_input in input_names, "Output name must be included in input names"
        self.output_shape_from_input = output_shape_from_input
        self.output_shape_idx = input_names.index(output_shape_from_input)
        
        self.embedder = Embedder(input_dims, patch_dims, dim, patch_emb=patch_emb)

        self.num_patches = 1 if pool == "cls" else 0
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

        self.pos_emb = pos_emb

        if pos_emb == 'satclip':
            if 'oci' in input_names:
                self.oci_embedding = nn.Parameter(torch.randn(1, self.num_input_patches['oci'], dim))
                self.oci_embedding = trunc_normal_(self.oci_embedding, std=.02)
            else:
                self.oci_embedding = None
            self.projector = nn.Linear(256, dim)
        else:
            self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))
            self.pos_embedding = trunc_normal_(self.pos_embedding, std=.02)

        self.num_register_tokens = num_register_tokens

        if num_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, dim))
            self.register_tokens = trunc_normal_(self.register_tokens, std=.02)
        else:
            self.register_tokens = None

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.pool = pool
        self.to_latent = nn.Identity()

        if self.pool == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
            self.output_shape = num_classes * input_dims[self.output_shape_idx][-2] * input_dims[self.output_shape_idx][-1]
            self.linear_head = nn.Linear(dim, self.output_shape)
        
        else:

            # calculate output shape per token
            token_output_shape = num_classes * patch_dims[self.output_shape_idx][-2] * patch_dims[self.output_shape_idx][-1]

            self.decoder = TokenDecoder(num_classes, *input_dims[self.output_shape_idx], *patch_dims[self.output_shape_idx], dim, pool=self.pool)

            self.linear_head = nn.Linear(dim, token_output_shape)

        self.init_weights()

    def init_weights(self):
        """ Initialize the weights in backbone """

        def _init_weights(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.Conv2d) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.Conv3d) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.LayerNorm) and m.elementwise_affine:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)            

    def forward(self, inputs, local_satclip=None, global_satclip=None, return_pos_embeddings=False):
        batch, device = inputs[0].shape[0], inputs[0].device

        x = self.embedder(inputs)
        if self.pool == "cls":
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        

        if self.pos_emb == 'satclip':
            # rearrange the satclip embeddings
            local_satclip = rearrange(local_satclip, "b h w c -> b (h w) c") # batch_size * 5 * 5 * 256 -> batch_size * 25 * 256
            global_satclip = rearrange(global_satclip, "b h w c -> b (h w) c") # batch_size * 12 * 6 * 256 -> batch_size * 72 * 256

            # append the pos embeddings
            if 'global' in self.input_names:
                pos_embeddings = torch.cat([local_satclip, global_satclip], dim=1) # batch_size * 97 * 256
            else:
                pos_embeddings = local_satclip

            # pass through the projector
            pos_embeddings = self.projector(pos_embeddings) # batch_size * 97 * dim

            # stack with the oci embeddings
            if self.oci_embedding is not None:
                all_pos_embeddings = torch.cat([pos_embeddings, self.oci_embedding.repeat(batch, 1, 1)], dim=1)
            else:
                all_pos_embeddings = pos_embeddings
            

            x += all_pos_embeddings
        else:
            x += self.pos_embedding.to(device, dtype=x.dtype)

        if self.register_tokens is not None:
            r = repeat(self.register_tokens, 'n d -> b n d', b = batch)
            x, ps = pack([x, r], 'b * d')
            x = self.transformer(x)
            x, _ = unpack(x, ps, 'b * d')
        else:
            x = self.transformer(x)

        if self.pool == "cls":
            x = x[:,0,:]
            x = self.to_latent(x)
            x = self.linear_head(x)
            x = rearrange(x, "b (nc h w) -> b nc h w", nc=self.num_classes, h=self.input_dims[self.output_shape_idx][-2], w=self.input_dims[self.output_shape_idx][-1])
        else:
            x = x[:,:self.num_input_patches[self.output_shape_from_input],:]
            x = self.decoder(x)

        if return_pos_embeddings:
            return x, all_pos_embeddings
        else:
            return x