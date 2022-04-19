# deeptrees

This was a set of small-scale experiments with applying *Tree Alternating Optimization* ([paper link](https://proceedings.neurips.cc/paper/2018/file/185c29dc24325934ee377cfda20e414c-Paper.pdf)) to the problem of learning a cascade of oblique decision trees. In this idea, we replace standard MLP layers in a neural network with oblique decision trees that have linear functions as leaves. We can then use standard SGD to update the weight matrices of the leaves, and use TAO to update the decision functions of each tree.

My conclusion is that this is not a very promising approach, since TAO doesn't have the same nice properties as SGD:

 1. In my implementation, TAO scales quadratically in the number of layers, whereas SGD is linear.
 2. The TAO paradigm does not easily apply to arbitrary computation graphs, and breaks down for parameter-sharing situations like convolution.

In my experiments, I found several interesting (but not very useful) observations:

 * Even a linear function can fit MNIST quite well.
 * The baseline of "0-depth trees with linear leaves" (basically just a linear network) is very hard to beat
 * The TAO-style update rule almost completely breaks for cases like convolution, where a tree can be applied multiple times to each input and the outputs can potentially interact. Even local approximations to TAO often fail to improve (or actually hurt) the loss on most iterations (e.g. replacing a single output value at a time, or using output gradients to approximate the loss function separately for each image patch).
 * If you fix the leaves randomly, and tree to only learn the decision function, the best accuracy you can get is typically something like 40% on MNIST with depth 3 trees. On the other hand, if you only use SGD on the leaves and fix the branches, you can easily get up to 97.8% accuracy.
 * Using policy gradients for learning decision functions in a tree works fairly well, and can actually work better than TAO for the case of convolution.
