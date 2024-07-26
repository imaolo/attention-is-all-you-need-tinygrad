from tinygrad import Tensor
from tinygrad.nn import Linear, LayerNorm, Embedding

# https://arxiv.org/abs/1706.03762

# FFN(x) = max(0, xW1 + b1)W2 + b2 
class FeedForward:
  def __init__(self, d_model, hidden_d_model):
    self.lin1 = Linear(d_model, hidden_d_model, bias=True)
    self.lin2 = Linear(hidden_d_model, d_model, bias=True)

  def __call__(self, x:Tensor) -> Tensor:
    return self.lin2(self.lin1(x).relu())

#Attention(Q, K, V ) = softmax(QKT√dk)V
def scaled_dot_product_attention(q: Tensor, k:Tensor, v:Tensor):
    return Tensor.softmax(q @ k.transpose(-2, -1) / (q.shape[-1]** 0.5)) @ v

# MultiHead(Q, K, V ) = Concat(head1, ..., headh)WO
#where headi = Attention(QWQi, KW Ki, V WVi)
class Attention:
    def __init__(self, d_model, n_heads):
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_d_model = d_model // n_heads

        assert self.head_d_model * n_heads == d_model, "d_model must be divisible by n_heads"

        self.v_lin = Linear(d_model, d_model)
        self.k_lin = Linear(d_model, d_model)
        self.q_lin = Linear(d_model, d_model)

        self.proj = Linear(d_model, d_model)

    def split_heads(self, x: Tensor, batch_size):
        return x.reshape(batch_size, self.n_heads, x.shape[-2], self.head_d_model)

    def __call__(self, v: Tensor, k:Tensor, q: Tensor):
        batch_size, seq_len = v.shape[0], v.shape[1]
        v = self.split_heads(self.v_lin(v), batch_size)
        k = self.split_heads(self.k_lin(k), batch_size)
        q = self.split_heads(self.q_lin(q), batch_size)

        attn = scaled_dot_product_attention(q, k, v).reshape(batch_size, seq_len, self.d_model)
        return self.proj(attn)

class Encoder:

    def __init__(self, d_model, n_heads, d_model_ff):
        self.attn = Attention(d_model, n_heads)
        self.norm1 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_model_ff)
        self.norm2 = LayerNorm(d_model)

    def __call__(self, x: Tensor):
        x = self.attn(x, x, x) + x
        x = self.norm1(x)
        x = self.ffn(x) + x
        return self.norm2(x)

class Decoder:

    def __init__(self, d_model, n_heads, d_model_ff):
        self.attn1 = Attention(d_model, n_heads)
        self.norm1 = LayerNorm(d_model)

        self.attn2 = Attention(d_model, n_heads)
        self.norm2 = LayerNorm(d_model)

        self.ffn = FeedForward(d_model, d_model_ff)
        self.norm3 = LayerNorm(d_model)

    def __call__(self, x: Tensor, enc_out:Tensor):
        x = self.attn1(x, x, x) + x
        x = self.norm1(x)

        x = self.attn2(x, enc_out, enc_out) + x
        x = self.norm2(x)

        x = self.ffn(x) + x
        return self.norm2(x)

# P E(pos,2i) = sin(pos/100002i/dmodel)
# P E(pos,2i+1) = cos(pos/100002i/dmodel)
class PositionalEncoding:
    
    def __init__(self, d_model, max_len):
        self.encoding = Tensor.zeros(max_len, d_model).contiguous()
        pos = Tensor.arange(0, max_len).unsqueeze(-1)
        div = 10000 ** (Tensor.arange(0, d_model, 2) / d_model)

        self.encoding[:, 0::2] = Tensor.sin(pos/div)
        self.encoding[:, 1::2] = Tensor.cos(pos/div)

        self.encoding = self.encoding.unsqueeze(0)
        self.encoding.requires_grad = False
    
    def __call__(self, x:Tensor):
        return x + self.encoding[:, :x.shape[1]]

class Transformer:

    def __init__(self, d_model, n_heads, d_ff, n_stacks, in_vocab_size, out_vocab_size, max_len):

        self.in_embed = Embedding(in_vocab_size, d_model)
        self.out_embed = Embedding(out_vocab_size, d_model)

        self.pos_embed = PositionalEncoding(d_model, max_len)

        self.encoders = [Encoder(d_model, n_heads, d_ff) for _ in range(n_stacks)]
        self.decoders = [Decoder(d_model, n_heads, d_ff) for _ in range(n_stacks)]

        self.proj = Linear(d_model, out_vocab_size)

    def __call__(self, input: Tensor, output:Tensor):

        input = self.in_embed(input)
        input = self.pos_embed(input)

        for encoder in self.encoders:
            input = encoder(input)

        output = self.out_embed(output)
        output = self.pos_embed(output)

        for decoder in self.decoders:
            output = decoder(output, input)

        return self.proj(output).softmax()

seq_len = 20
d_model = 20
batch_size = 20
n_heads = 4
d_ff = 512
n_stacks = 8
vocab_size = 20

# TODO - generation, masking, training

model = Transformer(d_model, n_heads, d_ff, n_stacks, vocab_size, vocab_size, seq_len)
model(Tensor.rand(1, seq_len), Tensor.rand(1, seq_len)).realize()