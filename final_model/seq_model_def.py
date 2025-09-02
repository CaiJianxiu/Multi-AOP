import torch
import torch.nn as nn

from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig, mLSTMBlockConfig
from xlstm import mLSTMLayerConfig, sLSTMBlockConfig, sLSTMLayerConfig, FeedForwardConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg = xLSTMBlockStackConfig(
    mlstm_block=mLSTMBlockConfig(
        mlstm=mLSTMLayerConfig(
            conv1d_kernel_size=4,
            qkv_proj_blocksize=8,
            num_heads=4
        )
    ),
    slstm_block=sLSTMBlockConfig(
        slstm=sLSTMLayerConfig(
            backend="cuda" if device.type == "cuda" else "cpu",
            num_heads=4,
            conv1d_kernel_size=4,
            bias_init="powerlaw_blockdependent",
        ),
        feedforward=FeedForwardConfig(
            proj_factor=2.0,
            act_fn="gelu",
            dropout=0.5
        ),
    ),
    context_length=256,
    num_blocks=3,
    embedding_dim=128,
    slstm_at= [1, 2]
    )
# xLSTM Component
class SequenceModel(nn.Module):
    def __init__(self, vocab_size=21, seq_length=50):
        super(SequenceModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, cfg.embedding_dim, padding_idx=0)
        self.xlstm_stack = xLSTMBlockStack(cfg)
        self.seq_length = seq_length
    def forward(self, x):
        x = self.embedding(x)
        x = self.xlstm_stack(x)
        return x


