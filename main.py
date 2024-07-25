from tinygrad import Tensor
from tinygrad.nn import Linear, LayerNorm

# https://arxiv.org/abs/1706.03762

# FFN(x) = max(0, xW1 + b1)W2 + b2 
class FeedForward:
  def __init__(self, dim, hidden_dim):
    self.lin1 = Linear(dim, hidden_dim, bias=True)
    self.lin2 = Linear(hidden_dim, dim, bias=True)

  def __call__(self, x:Tensor) -> Tensor:
    return self.lin2(self.lin1(x).relu())

#Attention(Q, K, V ) = softmax(QKT√dk)V
def scaled_dot_product_attention(q: Tensor, k:Tensor, v:Tensor):
    return Tensor.softmax(q@k.transpose(-2, -1) / (q.shape[-1]** 0.5)) @ v

# MultiHead(Q, K, V ) = Concat(head1, ..., headh)WO
#where headi = Attention(QWQi, KW Ki, V WVi)
class Attention:
    def __init__(self, dim, n_heads):
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        assert self.head_dim * n_heads == dim, "dim must be divisible by n_heads"

        self.v_lin = Linear(dim, dim)
        self.k_lin = Linear(dim, dim)
        self.q_lin = Linear(dim, dim)

        self.proj = Linear(dim, dim)

    def split_heads(self, x: Tensor, batch_size):
        return x.reshape(batch_size, self.n_heads, x.shape[-2], self.head_dim)

    def __call__(self, v: Tensor, k:Tensor, q: Tensor):
        batch_size = v.shape[0]
        v = self.split_heads(self.v_lin(v), batch_size)
        k = self.split_heads(self.k_lin(k), batch_size)
        q = self.split_heads(self.q_lin(q), batch_size)

        attn = scaled_dot_product_attention(q, k, v).reshape(batch_size, seq_len, self.dim)
        return self.proj(attn)


class TransformerBlock:

    def __init__(self, dim, n_heads, dim_ff):
        self.attn = Attention(dim, n_heads)
        self.norm1 = LayerNorm(dim)
        self.ffn = FeedForward(dim, dim_ff)
        self.norm2 = LayerNorm(dim)

    def __call__(self, x: Tensor):
        x = self.attn(x, x, x) + x
        x = self.norm1(x)
        x = self.ffn(x) + x
        return self.norm2(x)

class Transformer:

    def __init__(self):
        # TODO
        pass
   


seq_len = 20
dim = 20
batch_size = 20
n_heads = 4
d_ff = 512

t_block = TransformerBlock(dim, n_heads, d_ff)
t_block(Tensor.rand(batch_size, seq_len, dim)).realize()






