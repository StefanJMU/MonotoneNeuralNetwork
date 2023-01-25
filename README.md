# MonotoneNeuralNetwork

## Structure

Compact neural network architecture approximating monotonic functions.
The restriction to monotonical functions is achieved by small changes in the general
formulation of a feed-forward architecture:

Considering a single layer network $f$ with input $x \in \mathbb{R}^n$ and $M \in \\{diag(k) : k \in \\{-1, 1\\}\\}^n$, indicating
that the network outputs are required to be non-increasing (-1) or non-decreasing (1) for the respective element
of $x$, the output of the layer is defined as 

$f(x) = \sigma(W^2Mx + b)$

, for any monotonically increasing activation function $\sigma$.

The partial derivative is then

$\frac{\delta f(x)_i}{\delta x_j} = \dot{\sigma}(W^2Mx + b)_i M\_{jj} W\_{ij}^2$

, with $sign(\frac{\delta f(x)_i}{\delta x_j}) = sign(M\_{jj})$.

Setting in a multi-layer network for all non-input layers $M = diag(\\{1, \dots, 1\\})$, 
the required monotonic property is preserved in the layer composition.

## Use cases

Regression with monotonic constraints (isotonic regression) can be a useful way of incorporating
a priori knowledge of a problem domain into the regression approach, in order to avoid semantically degenerative fitting.
Using a neural network structure allows the end-to-end learning of monotonic mappings between
data ranges as subcomponent in larger neural network models.

![there should be an image](monotone.png "Regression of monotone function")
![there should be an image](not_monotone.png "Regression of an non-monotone function")



