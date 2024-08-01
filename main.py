import transformer_pt, transformer_tg
from tinygrad.helpers import Timing

seq_len = 4
pred_factor = 2
bound_factor = 5
vocab_size = seq_len * pred_factor * bound_factor
d_model = 50
n_heads = 6
d_ff = 512
n_stacks = 6
num_loops = 1000

model = transformer_pt.Transformer(d_model, n_heads, d_ff, n_stacks, vocab_size, seq_len)
with Timing("pytorch time: "):
    loss = transformer_pt.train(model, pred_factor, bound_factor, num_loops)
print("pytorch loss: ", loss)

model = transformer_tg.Transformer(d_model, n_heads, d_ff, n_stacks, vocab_size, seq_len)
with Timing("tinygrad time: "):
    loss = transformer_tg.train(model, pred_factor, bound_factor, num_loops)
print("tinygrad loss: ", loss)