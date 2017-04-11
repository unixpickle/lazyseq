# lazyrnn

This is an API for evaluating and training RNNs on memory-constrained systems. It provides APIs for some of the memory-saving algorithms described in [Gruslys et al.](https://arxiv.org/abs/1606.03401) and [Chen et al.](https://arxiv.org/abs/1604.06174).

# Why?

Traditional back-propagation requires O(T) memory for sequences of length T. This makes RNNs difficult to train on certain tasks, even when the RNN *could* theoretically learn the task. One example of such a task is meta-learning with long episodes.

With less naive back-propagation techniques, memory consumption can be reduced to O(log(T)) without significant performance sacrifices. This makes RNNs suitable for a much wider range of applications.
