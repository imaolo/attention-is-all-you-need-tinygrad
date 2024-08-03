Benchmark the transformer architecture from the paper "Attention Is All You Need" in pytorch and tinygrad. Written for M-Series macbooks.

Using torch device 'mps' is a lot slower than 'cpu'

Using 'METAL' for tinygrad fails because a kernel is generated which has too many arguments. Use 'GPU' instead.
