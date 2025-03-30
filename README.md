## Expected Return Symmetries
ICLR 2025
Singapore

[arXiv (arXiv:2502.01711)](https://arxiv.org/abs/2502.01711)

### Paper TL;DR

We define the group of *expected return symmetries*, a group of transformations whose action on policies preserves their expected return. This group of symmetries is a superset of the standard group typically considered whose action over the MDP preserves dynamics (i.e. transition, reward probabilities, etc.)

In particular, we focus on the application of expected return symmetries in decentralized POMDPs, a useful abstraction for cooperative settings. To this end, we find that expected return symmetries improve coordination outcomes of agents across a variety of enviornments. Not only do expected return symmetries improve coordination outcomes significantly more than the standard, dynamics preserving group, they also:
* can be learned through reward maximization, which is often easier to optimize for than dynamics preservation;
* can be applied in environments where the dynamics preserving group only contains the trivial, identity transformation.

### Reproducing Results

We recommend installing the dependencies in `requirements.txt` and using the provided Dockerfile.

For Hanabi and Overcooked V2, we provide some sample pre-trained self-play optimal joint policies.