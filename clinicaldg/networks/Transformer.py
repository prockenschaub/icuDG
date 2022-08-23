import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Self-attention network -------------------------------------------------------
#
# Implementation of Multi Head Attention as described in Vaswani et al. (2017) 
# Attention is all you need (https://arxiv.org/abs/1706.03762)
#
# Code copied from https://github.com/ratschlab/HIRID-ICU-Benchmark/


def parrallel_recomb(q_t, kv_t, att_type='all', local_context=3, bin_size=None):
    """ Return mask of attention matrix (ts_q, ts_kv) """
    with torch.no_grad():
        q_t[q_t == -1.0] = float('inf')  # We want padded to attend to everyone to avoid any nan.
        kv_t[kv_t == -1.0] = float('inf')  # We want no one to attend the padded values

        if bin_size is not None:  # General case where we use unaligned timesteps.
            q_t = q_t / bin_size
            starts_q = q_t[:, 0:1].clone()  # Needed because of Memory allocation issue
            q_t -= starts_q
            kv_t = kv_t / bin_size
            starts_kv = kv_t[:, 0:1].clone()  # Needed because of Memory allocation issue
            kv_t -= starts_kv

        bs, ts_q = q_t.size()
        _, ts_kv = kv_t.size()
        q_t_rep = q_t.view(bs, ts_q, 1).repeat(1, 1, ts_kv)
        kv_t_rep = kv_t.view(bs, 1, ts_kv).repeat(1, ts_q, 1)
        diff_mask = (q_t_rep - kv_t_rep).to(q_t_rep.device)
        if att_type == 'all':
            return (diff_mask >= 0).float()
        if att_type == 'local':
            return ((diff_mask >= 0) * (diff_mask <= local_context) + (diff_mask == float('inf'))).float()
        if att_type == 'strided':
            return ((diff_mask >= 0) * (torch.floor(diff_mask) % local_context == 0) + (
                    diff_mask == float('inf'))).float()


class PositionalEncoding(nn.Module):
    "Positiona Encoding, mostly from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html"

    def __init__(self, emb, max_len=3000):
        super().__init__()
        pe = torch.zeros(max_len, emb)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb, 2).float() * (-math.log(10000.0) / emb))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        bs, n, emb = x.size()
        return x + self.pe[:, :n, :]


class SelfAttention(nn.Module):
    """Multi Head Attention block from Attention is All You Need.
    Input has shape (batch_size, n_timestemps, emb).

    ----------
    emb:
        Dimension of the input vector.
    hidden:
        Dimension of query, key, value matrixes.
    heads:
        Number of heads.

    mask:
        Mask the future timestemps
    """

    def __init__(self, emb, hidden, heads=8, mask=True, att_type='all', local_context=None, mask_aggregation='union',
                 dropout_att=0.0):
        """Initialize the Multi Head Block."""
        super().__init__()

        self.emb = emb
        self.heads = heads
        self.hidden = hidden
        self.mask = mask
        self.drop_att = nn.Dropout(dropout_att)

        # Sparse transformer specific params
        self.att_type = att_type
        self.local_context = local_context
        self.mask_aggregation = mask_aggregation

        # Query, keys and value matrices
        self.w_keys = nn.Linear(emb, hidden * heads, bias=False)
        self.w_queries = nn.Linear(emb, hidden * heads, bias=False)
        self.w_values = nn.Linear(emb, hidden * heads, bias=False)

        # Output linear function
        self.unifyheads = nn.Linear(heads * hidden, emb)

    def forward(self, x):
        """
        x:
            Input data tensor with shape (batch_size, n_timestemps, emb)
        hidden:
            Hidden dim (dimension of query, key, value matrixes)

        Returns
            Self attention tensor with shape (batch_size, n_timestemps, emb)
        """
        # bs - batch_size, n - vectors number, emb - embedding dimensionality
        bs, n, emb = x.size()
        h = self.heads
        hidden = self.hidden

        keys = self.w_keys(x).view(bs, n, h, hidden)
        queries = self.w_queries(x).view(bs, n, h, hidden)
        values = self.w_values(x).view(bs, n, h, hidden)

        # fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(bs * h, n, hidden)
        queries = queries.transpose(1, 2).contiguous().view(bs * h, n, hidden)
        values = values.transpose(1, 2).contiguous().view(bs * h, n, hidden)

        # dive on the square oot of dimensionality
        queries = queries / (hidden ** (1 / 2))
        keys = keys / (hidden ** (1 / 2))

        # dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))

        if self.mask:  # We deal with different masking and recombination types here
            if isinstance(self.att_type, list):  # Local and sparse attention
                if self.mask_aggregation == 'union':
                    mask_tensor = 0
                    for att_type in self.att_type:
                        mask_tensor += \
                        parrallel_recomb(torch.arange(1, n + 1, dtype=torch.float, device=dot.device).reshape(1, -1),
                                         torch.arange(1, n + 1, dtype=torch.float, device=dot.device).reshape(1, -1),
                                         att_type,
                                         self.local_context)[0]
                    mask_tensor = torch.clamp(mask_tensor, 0, 1)
                    dot = torch.where(mask_tensor.bool(),
                                      dot,
                                      torch.tensor(float('-inf')).to(dot.device)).view(bs * h, n, n)

                elif self.mask_aggregation == 'split':

                    dot_list = list(torch.split(dot, dot.shape[0] // len(self.att_type), dim=0))
                    for i, att_type in enumerate(self.att_type):
                        mask_tensor = \
                        parrallel_recomb(torch.arange(1, n + 1, dtype=torch.float, device=dot.device).reshape(1, -1),
                                         torch.arange(1, n + 1, dtype=torch.float, device=dot.device).reshape(1, -1),
                                         att_type,
                                         self.local_context)[0]

                        dot_list[i] = torch.where(mask_tensor.bool(), dot_list[i],
                                                  torch.tensor(float('-inf')).to(dot.device)).view(*dot_list[i].shape)
                    dot = torch.cat(dot_list, dim=0)
            else:  # Full causal masking
                mask_tensor = \
                parrallel_recomb(torch.arange(1, n + 1, dtype=torch.float, device=dot.device).reshape(1, -1),
                                 torch.arange(1, n + 1, dtype=torch.float, device=dot.device).reshape(1, -1),
                                 self.att_type,
                                 self.local_context)[0]
                dot = torch.where(mask_tensor.bool(),
                                  dot,
                                  torch.tensor(float('-inf')).to(dot.device)).view(bs * h, n, n)

        # dot now has row-wise self-attention probabilities
        dot = F.softmax(dot, dim=2)

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(bs, h, n, hidden)

        # apply the dropout
        out = self.drop_att(out)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(bs, n, h * hidden)
        return self.unifyheads(out)


class TransformerBlock(nn.Module):

    def __init__(self, emb, hidden, heads, ff_hidden_mult, dropout=0.0, mask=True, dropout_att=0.0):
        super().__init__()

        self.attention = SelfAttention(emb, hidden, heads=heads, mask=mask, dropout_att=dropout_att)
        self.mask = mask
        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        attended = self.attention.forward(x)
        x = self.norm1(attended + x)
        x = self.drop(x)
        fedforward = self.ff(x)
        x = self.norm2(fedforward + x)

        x = self.drop(x)

        return x


class Transformer(nn.Module):
    def __init__(self, emb, hidden, heads, ff_hidden_mult, depth, dropout=0.0, l1_reg=0,
                 pos_encoding=True, dropout_att=0.0):
        super().__init__()

        self.input_embedding = nn.Linear(emb, hidden)  # This acts as a time-distributed layer by defaults
        if pos_encoding:
            self.pos_encoder = PositionalEncoding(hidden)
        else:
            self.pos_encoder = None

        tblocks = []
        for i in range(depth):
            tblocks.append(TransformerBlock(emb=hidden, hidden=hidden, heads=heads, mask=True,
                                            ff_hidden_mult=ff_hidden_mult,
                                            dropout=dropout, dropout_att=dropout_att))

        self.tblocks = nn.Sequential(*tblocks)
        self.l1_reg = l1_reg

    def forward(self, x):
        x = self.input_embedding(x)
        if self.pos_encoder is not None:
            x = self.pos_encoder(x)
        x = self.tblocks(x)

        return x
