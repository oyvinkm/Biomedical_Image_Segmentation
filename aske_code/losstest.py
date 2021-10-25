import torch
import torch.nn.functional as F
from loss import _BCEWithLogitsLoss

inp = torch.randn(1, requires_grad=True)
tar = torch.empty(1).random_(2)

loss = F.binary_cross_entropy_with_logits(inp,tar)
lossman = _BCEWithLogitsLoss()

print('bce loss')
print(loss)
print('manual loss')
print(lossman(inp,tar))