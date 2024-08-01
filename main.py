from tinygrad import Tensor, nn, dtypes, TinyJit, Device
import math

# METAL is broken
Device.DEFAULT = 'METAL'

# https://arxiv.org/abs/1706.03762

# FFN(x) = max(0, xW1 + b1)W2 + b2 
class FeedForward:
  def __init__(self, d_model, hidden_d_model):
    self.lin1 = nn.Linear(d_model, hidden_d_model)
    self.lin2 = nn.Linear(hidden_d_model, d_model)

  def __call__(self, x:Tensor) -> Tensor:
    return self.lin2(self.lin1(x).relu())

# FeedForward with dropout, residual, and norm
class FeedForwardLayer(FeedForward):
    def __init__(self, d_model, hidden_d_model, dropout=0.1):
        super().__init__(d_model, hidden_d_model)
        self.dropout = dropout
        self.norm = nn.LayerNorm(d_model)
    
    def __call__(self, x):
       return self.norm(super().__call__(x).dropout(self.dropout) + x)

# MultiHead(Q, K, V ) = Concat(head1, ..., headh)WO
#where headi = Attention(QWQi, KW Ki, V WVi)
class MultiHeadAttention():
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, dropout=0.1):
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout

        self.d_k = d_k if d_k is not None else d_model // n_heads
        self.d_v = d_v if d_v is not None else d_model // n_heads
        self.output_dim = self.d_v * n_heads

        self.k_lin = nn.Linear(d_model, n_heads * self.d_k)
        self.q_lin = nn.Linear(d_model, n_heads * self.d_k)
        self.v_lin = nn.Linear(d_model, n_heads * self.d_v)
        self.fc = nn.Linear(self.output_dim, d_model)

    def __call__(self, q, k, v, mask=None) -> Tensor:
        batch_size = v.shape[0]

        # linear layers and split heads
        q = self.q_lin(q).reshape(batch_size, self.n_heads, -1, self.d_k)
        k = self.k_lin(k).reshape(batch_size, self.n_heads, -1, self.d_k)
        v = self.v_lin(v).reshape(batch_size, self.n_heads, -1, self.d_v)

        # scaled attention, concatenate heads, and fully connected layer
        return self.fc(q.scaled_dot_product_attention(k, v, mask, self.dropout).reshape(batch_size, -1, self.output_dim))

# MultiHeadAttention with dropout, residual, and norm
class MultiHeadAttentionLayer(MultiHeadAttention):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, dropout=0.1):
        super().__init__(d_model, n_heads, d_k, d_v)
        self.dropout = dropout
        self.norm = nn.LayerNorm(d_model)

    def __call__(self, v, k, q, mask) -> Tensor:
       return self.norm(super().__call__(k, q, v, mask).dropout(self.dropout) + q)

# P E(pos,2i) = sin(pos/100002i/dmodel)
# P E(pos,2i+1) = cos(pos/100002i/dmodel)
class PositionalEncoding:
    def __init__(self, d_model, max_len):
        super().__init__()
        self.encoding = Tensor.zeros(max_len, d_model).contiguous()
        pos = Tensor.arange(0, max_len).unsqueeze(1)
        div_term = Tensor.exp(Tensor.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = Tensor.sin(pos * div_term)
        self.encoding[:, 1::2] = Tensor.cos(pos * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def __call__(self, x):
        return x + self.encoding[:, :x.shape[1]]

class Encoder():
    def __init__(self, d_model, n_heads, d_model_ff, d_k=None, d_v=None):
        self.attn = MultiHeadAttentionLayer(d_model, n_heads, d_k, d_v)
        self.ffn = FeedForwardLayer(d_model, d_model_ff)

    def __call__(self, x, mask):
        return self.ffn(self.attn(x, x, x, mask))
class Decoder:
    def __init__(self, d_model, n_heads, d_model_ff, d_k=None, d_v=None):
        self.attn1 = MultiHeadAttentionLayer(d_model, n_heads, d_k, d_v)
        self.attn2 = MultiHeadAttentionLayer(d_model, n_heads, d_k, d_v)
        self.ffn = FeedForwardLayer(d_model, d_model_ff)

    def __call__(self, x, enc_out, mask):
        x = self.attn1(x, x, x, mask)
        x = self.attn2(x, enc_out, enc_out, mask)
        return self.ffn(x)

class Transformer:
    def __init__(self, d_model, n_heads, d_ff, n_layers, vocab_size, max_len, d_k=None, d_v=None):
        super().__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.in_embed = nn.Embedding(vocab_size, d_model)
        self.out_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = PositionalEncoding(d_model, max_len)
        self.encoders = [Encoder(d_model, n_heads, d_ff, d_k, d_v) for _ in range(n_layers)]
        self.decoders = [Decoder(d_model, n_heads, d_ff, d_k, d_v) for _ in range(n_layers)]
        self.proj = nn.Linear(d_model, vocab_size)

    def __call__(self, enc_in, dec_in):
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

        return self.proj(dec_out).log_softmax()
    
def train(model: Transformer, pred_factor:int, bound_factor:int, num_loops:int) -> int:
    X_TRAIN: Tensor = Tensor.arange(0, model.max_len).unsqueeze(0).cast(dtypes.int)
    opt = nn.optim.Adam(nn.state.get_parameters(model))
    empty_tensor = Tensor.zeros(model.max_len).cast(dtypes.int).unsqueeze(0)

    @TinyJit
    def train_step(x: Tensor, y: Tensor) -> Tensor:
        with Tensor.train():
            opt.zero_grad()
            outputs = model(x, empty_tensor)
            loss = outputs.sparse_categorical_crossentropy(y)
            loss.backward()
            opt.step()
            return loss.realize()

    # Training loop
    limit_size = (model.vocab_size//((model.max_len - 1)*pred_factor)) - bound_factor
    for i in range(1, num_loops):
        train_scale = (i%limit_size)+1
        x_train = (X_TRAIN * train_scale).cast(dtypes.int64)
        y_train = (x_train * pred_factor).float()
        loss = train_step(x_train.realize(), y_train.realize())
    return loss.item()

seq_len = 4
pred_factor = 2
bound_factor = 5
vocab_size = seq_len * pred_factor * bound_factor
d_model = 50
n_heads = 6
d_ff = 512
n_stacks = 6
num_loops = 100

model = Transformer(d_model, n_heads, d_ff, n_stacks, vocab_size, seq_len)
print(train(model, pred_factor, bound_factor, num_loops))