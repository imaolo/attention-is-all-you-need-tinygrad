import torch, math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# https://arxiv.org/abs/1706.03762

# FFN(x) = max(0, xW1 + b1)W2 + b2
class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_d_model):
        super().__init__()
        self.lin1 = nn.Linear(d_model, hidden_d_model)
        self.lin2 = nn.Linear(hidden_d_model, d_model)

    def forward(self, x):
        return self.lin2(F.relu(self.lin1(x)))

# FeedForward with dropout, residual, and norm
class FeedForwardLayer(FeedForward):
    def __init__(self, d_model, hidden_d_model, dropout=0.1):
        super().__init__(d_model, hidden_d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
       return self.norm(self.dropout(super().forward(x)) + x)

# MultiHead(Q, K, V ) = Concat(head1, ..., headh)WO
#where headi = Attention(QWQi, KW Ki, V WVi)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout_p = dropout

        self.d_k = d_k if d_k is not None else d_model // n_heads
        self.d_v = d_v if d_v is not None else d_model // n_heads
        self.output_dim = self.d_v * n_heads

        self.k_lin = nn.Linear(d_model, n_heads * self.d_k)
        self.q_lin = nn.Linear(d_model, n_heads * self.d_k)
        self.v_lin = nn.Linear(d_model, n_heads * self.d_v)
        self.fc = nn.Linear(self.output_dim, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = v.size(0)

        # linear layers and split heads
        q = self.q_lin(q).view(batch_size, self.n_heads, -1, self.d_k)
        k = self.k_lin(k).view(batch_size, self.n_heads, -1, self.d_k)
        v = self.v_lin(v).view(batch_size, self.n_heads, -1, self.d_v)

        # scaled attention, concatenate heads, and fully connected layer
        return self.fc(F.scaled_dot_product_attention(q, k, v, mask, self.dropout_p).reshape(batch_size, -1, self.output_dim))

# MultiHeadAttention with dropout, residual, and norm
class MultiHeadAttentionLayer(MultiHeadAttention):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, dropout=0.1):
        super().__init__(d_model, n_heads, d_k, d_v)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, v, k, q, mask):
       return self.norm(self.dropout(super().forward(k, q, v, mask)) + q)

# P E(pos,2i) = sin(pos/10000^(2i/dmodel))
# P E(pos,2i+1) = cos(pos/10000^(2i/dmodel))
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        self.encoding = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(pos * div_term)
        self.encoding[:, 1::2] = torch.cos(pos * div_term)
        self.encoding = self.encoding.unsqueeze(0)
        self.register_buffer('pe', self.encoding)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, d_model_ff, d_k=None, d_v=None):
        super().__init__()
        self.attn = MultiHeadAttentionLayer(d_model, n_heads, d_k, d_v)
        self.ffn = FeedForwardLayer(d_model, d_model_ff)

    def forward(self, x, mask):
        return self.ffn(self.attn(x, x, x, mask))

class Decoder(nn.Module):
    def __init__(self, d_model, n_heads, d_model_ff, d_k=None, d_v=None):
        super().__init__()
        self.attn1 = MultiHeadAttentionLayer(d_model, n_heads, d_k, d_v)
        self.attn2 = MultiHeadAttentionLayer(d_model, n_heads, d_k, d_v)
        self.ffn = FeedForwardLayer(d_model, d_model_ff)

    def forward(self, x, enc_out, mask):
        x = self.attn1(x, x, x, mask)
        x = self.attn2(x, enc_out, enc_out, mask)
        return self.ffn(x)

class Transformer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers, vocab_size, max_len, d_k=None, d_v=None):
        super().__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.in_embed = nn.Embedding(vocab_size, d_model)
        self.out_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = PositionalEncoding(d_model, max_len)
        self.encoders = nn.ModuleList([Encoder(d_model, n_heads, d_ff, d_k, d_v) for _ in range(n_layers)])
        self.decoders = nn.ModuleList([Decoder(d_model, n_heads, d_ff, d_k, d_v) for _ in range(n_layers)])
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, enc_in, dec_in):
        enc_in = self.in_embed(enc_in)
        enc_in = self.pos_embed(enc_in)

        for encoder in self.encoders:
            enc_in = encoder(enc_in, None)
        enc_out = enc_in

        dec_in = self.out_embed(dec_in)
        dec_in = self.pos_embed(dec_in)

        for decoder in self.decoders:
            dec_in = decoder(dec_in, enc_out, None)
        dec_out = dec_in

        return F.log_softmax(self.proj(dec_out), dim=-1)

# mps is slower
device = torch.device('cpu')

def train(model: Transformer, pred_factor:int, bound_factor:int, num_loops:int) -> int:
    X_TRAIN = torch.arange(0, model.max_len).int().unsqueeze(0).long()
    opt = optim.Adam(model.parameters())
    empty_tensor = torch.zeros(model.max_len).int().unsqueeze(0).to(device)
    lossfunc = nn.CrossEntropyLoss()
    model.to(device)

    def train_step(x, y):
        opt.zero_grad()
        outputs = model(x, empty_tensor)
        loss: torch.Tensor = lossfunc(outputs.view(model.max_len, -1), y.view(-1))
        loss.backward()
        opt.step()
        return loss
    model.train()
    limit_size = (model.vocab_size//((model.max_len - 1)*pred_factor)) - bound_factor
    for i in range(1, num_loops):
        train_scale = (i%limit_size)+1
        x_train = (X_TRAIN * train_scale).long().to(device)
        y_train = (x_train * pred_factor).long().to(device)
        loss = train_step(x_train, y_train)
    return loss.item()