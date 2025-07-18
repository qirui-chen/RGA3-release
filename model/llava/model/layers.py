import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x



class MaskPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask):

        mask = F.interpolate(mask, size=(24, 24), mode='bilinear', align_corners=False)

        # Maybe flatten mask

        b, t, l, d = x.shape
        b, t, h, w = mask.shape

        mask = mask.flatten(2) # b, t, l
        mask = (mask > 0).to(x.dtype)
        denorm = mask.sum(dim=-1, keepdim=True) + 1e-8

        # (bt)ld -> (bt)ld, (bt)l1 -> (bt)d
        mask_pooled_x = torch.einsum(
            "btld,btl->btd",
            x,
            mask / denorm,
        )
        return mask_pooled_x