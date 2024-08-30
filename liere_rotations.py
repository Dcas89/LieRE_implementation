import torch
from collections.abc import Iterable
import torch.nn as nn
import math
from einops import rearrange


class PositionEncoderBase(nn.Module):
    def __init__(self, image_size, patch_size, input_dimensionality):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        if not isinstance(self.patch_size, Iterable):
            self.patch_size = [self.patch_size] * input_dimensionality
        if not isinstance(self.image_size, Iterable):
            self.image_size = [self.image_size] * input_dimensionality

    def forward(self, image_sizes: torch.Tensor):
        assert (
            image_sizes.shape[0] == 1
        )  # only support one image size for the batch for now
        image_size = image_sizes.tolist()[0]
        steps_per_axis = (
            math.ceil(dim_size / patch_dim)
            for dim_size, patch_dim in zip(image_size, self.patch_size)
        )

        normalized_positions = torch.cartesian_prod(
            *(
                torch.linspace(0, 1, steps, device=image_sizes.device)
                for steps in steps_per_axis
            )
        )
        return normalized_positions.unsqueeze(0)


class LierePositionEncoder(PositionEncoderBase):
    def __init__(
        self,
        image_size,
        patch_size,
        dim,
        heads,
        input_dimensionality,
    ):
        super().__init__(image_size, patch_size, input_dimensionality)
        self.input_dimensionality = input_dimensionality
        self.head_dim = dim // heads

        # initialize the generator parameters
        # https://github.com/naver-ai/rope-vit/blob/c6aa201ee795daa4f841e2f9585164bb23a0b819/deit/models_v2_rope.py#L150C13-L150C76

        self.generator_raw_params = nn.Parameter(
            torch.rand(
                input_dimensionality,
                1, # Replace with proper value if you want to use a block-diagonal generator
                self.head_dim,
                self.head_dim,
            ) *
            match.pi * 2 # RoPE-Mixed scaled by 2 pi, scaling by a constant https://github.com/naver-ai/rope-vit/blob/c6aa201ee795daa4f841e2f9585164bb23a0b819/deit/models_v2_rope.py#L25
        )

    def forward(self, image_sizes: torch.Tensor, dtype):

        # Shape: [bs, num_tokens, dimensionality]
        positions = super().forward(image_sizes)

        # Shape: [generator_repeats, input_dimensionality, num_generators, generator_dim, generator_dim]
        upper_triangle = torch.triu(self.generator_raw_params, diagonal=1) - 0.5

        # Shape: [generator_repeats, input_dimensionality, num_generators, generator_dim, generator_dim]
        skew_bases = upper_triangle - torch.transpose(upper_triangle, -1, -2)

        # Shape: [bs, num_tokens, dimensionality]
        in_basis_positions = (
            positions.reshape(list(positions.shape) + [1] * 3) * skew_bases
        )

        generator_pos = torch.sum(in_basis_positions, dim=-4)  # sum over dimensions

        # add an identity for the CLS token.
        cls_generator = torch.zeros_like(generator_pos[:, 0, ...]).unsqueeze(1)
        generator_pos = torch.cat((cls_generator, generator_pos), dim=1)

        return torch.matrix_exp(generator_pos.to(dtype=torch.float32)).to(
            dtype=positions.dtype
        )


class FlexibleAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5
        self.dim_head = dim_head

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def apply_transforms(self, x, positional_transforms):
        generator_dim = positional_transforms.shape[-1]
        batch_size, num_heads, num_tokens, head_size = x.shape
        num_rotators = positional_transforms.shape[-3]
        # ipdb.set_trace()
        rotatable_dim = generator_dim * num_rotators

        assert head_size == self.dim_head, "Head dims and head size have to be the same"
        # Shape: [batch size, heads_num, tokens_num, head_dim]
        rotatable_states = x[..., :rotatable_dim]
        unrotatable_states = x[...,  rotatable_dim:]
        states_split = rotatable_states.reshape(
            (
                batch_size,
                num_heads,
                num_tokens,
                num_rotators,
                generator_dim,
                1,
            )
        )
        # Shape: [batch, num_rotator , 65, heads_num, generator_dim, generator_dim]
        positional_transforms = positional_transforms.reshape(
            (
                1,
                num_tokens,
                num_rotators,
                generator_dim,
                generator_dim,
            )
        )
        # why unsqueeze?
        rotated_states = torch.matmul(positional_transforms, states_split)
        return torch.cat(
            [rotated_states.flatten(start_dim=-3), unrotatable_states], axis=-1
        )

    def forward(self, x, positional_transforms=None):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        if positional_transforms is not None:
            # k is transformed in the next step.

            q, k = self.apply_transforms(
                q, positional_transforms
            ), self.apply_transforms(k, positional_transforms)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)

if __name__ == '__main__':
    position_encoder = LierePositionEncoder((224,224), (16,16), 128, 8, 2)
    attn = FlexibleAttention(128, 8, 128//8, 0)
    
    img_sizes = torch.tensor([[224,224]])
    position_encodings = position_encoder(img_sizes, torch.float32)
    
    fake_tokens = torch.rand((2, 1 + (224//16)**2, 128)) # 1 is coming from the CLS token
    
    print(fake_tokens.shape)
    print(f'Positional transforms shape {position_encodings.shape}')
    embeddings = attn(fake_tokens, position_encodings)
    
    assert fake_tokens.shape == embeddings.shape
    
    print("Done")
