from torch.accelerator import current_accelerator, is_available

HIDDEN_SIZE  = 64
EPOCH_SIZE   = 24
LR           = 0.001
DEVICE       = current_accelerator().type if is_available() else 'cpu'