### Expected Return Symmetry Training Scripts

The folder contains Pure Jax IPPO implementations based on [JaxMARL](https://github.com/FLAIROx/JaxMARL/tree/main).

`ippo_ff_hanabi.py` trains a joint policy parameterized by a FF network with self-play.

`group_ippo_ff_hanabi_neural_symm_disc.py` trains an expected return symmetry, taking pre-trained self-play joint policies as input. The expected return symmetry can optionally be regularized to be compositional with itself.

`group_ippo_ff_hanabi_neural_symm_disc.py` trains an expected return symmetry regularized with compositionality (with pre-trained expected return symmetries, given as input) and invertibility (via a L1 reconstruction loss).
