
# TrainStep:

TrainStep, with the simplest terms, is actually a function to manage one step of model training. First, it performs forward pass to compute logits and loss, and apply backpropagation for gradient calculation. Then, by using computed gradients, it updates model parameters. These three operations can be regarded as the primitive and obligatory part of one training step. At this point, we can extend the functionality of TrainStep by adding the computation of evaluation metrics and updating batch-norm statistics. You can find a simple code example below. For remainder, you can look at [previous tutorial](https://github.com/GoktugGuvercin/Flax-Tutorials/tree/main/TrainState)

```py
import jax
import optax

softmax_ce = optax.softmax_cross_entropy

def train_step(state, batch):
  
  def loss_fn(params):
    x, y = batch["images"], batch["labels"]
    logits = state.apply_fn({"params": params}, x=x) # forward pass
    loss = softmax_ce(logits, y)
    return loss, logits

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grads = grad_fn(state.params) # backward pass
  state = state.apply_gradients(grads)
```


## Questions and Answers:

> *Should we define TrainStep as a function ?*
* TrainStep does not have to be defined as a function. It is also possible to design it as a class like an interface.

$$\$$
> *Why do we define loss function inside `train_step()` ?*
* We define loss function inside `train_step()` so that it can directly access *state* object. Otherwise, `loss_fn()` has to take *state* object as an input argument. In this case, we need to specify, by using `argnum=0`, for which input argument (params or state), the gradient of `loss_fn()` will be computed:

$$\$$ 
> *Why don't we use `jax.grad()`?*
* We can use `jax.grad()` operator, but `jax.value_and_grad()` performs forward and backward passes at the same time. The latter is more code-efficient.

$$\$$ 
> *Why is forward pass applied inside loss function ? Sould loss function just compute loss value ?*
* JAX is capable of computing the gradients of any function with respect to its input arguments. To be able to propogate the gradients from computed loss to logits and then from logits to model parameters, forward pass and calculation of loss need to be in same function. As an alternative way, these operations can be isolated to two different functions, but in that case they need to be called inside an another function for full gradient flow without any interruption:

$$\$$
> *The object "state" contains the data member "params" and `loss_fn()` has direct access to "state" object. So, why are we passing params as input argument to loss function ?*
* Yes, `loss_fn()` can reach *"params"* by means of *"state"* object directly.So, passing it as input argument seems to be redundant. However, if we do not pass it, we cannot differentiate the loss function. JAX takes the derivative of functions with respect to their input arguments. 

## TrainStep Interface:

As specified above, TrainStep can be also implemented as an interface to control the training stage. In that way, it can be more useful and understandle, because even though its classical function implementation given above seems to be shorter, it is more complicated. This causes Flax-users to have so many different questions about the structure of code. That is why understanding and implementing TrainState in class-interface is more recommended, one of which is given below:

```py
import jax
import flax
import optax
from flax.training.train_state import TrainState

class TrainStep:
    def __init__(self, state: TrainState):
        self.state = state
        self.softmax_ce_fn = optax.softmax_cross_entropy
        self.grad_fn = jax.value_and_grad(self.loss_fn, argnums=0, has_aux=True)

    # backward() in torch
    def backward(self, batch):
        return self.grad_fn(self.state.params, batch)

    # optim.step() in torch
    def train_step(self, batch):
        (loss, (logits, updates)), grads = self.backward(batch)
        self.state = self.state.apply_gradients(grads=grads)
        self.state = self.state.replace(batch_stats=updates["batch_stats"])
        # metric computation

    # model() + loss_fn() in torch
    def loss_fn(self, params, batch):
        imgs, labels = batch["image"], batch["labels"]
        variables = {'params': params, 'batch_stats': self.state.batch_stats}

        logits, updates = self.state.apply_fn(variables, x=batch['image'],
                                              train=True, mutable=['batch_stats'])
        loss = self.softmax_ce_fn(logits, labels)
        return loss, (logits, updates)

```

#### Class Initializer:
As you know, all necessary data members and the function to apply forward pass are collected in TrainState object, so managing train steps by using it is inevitable. In addition, for the computation of final loss and network gradients, we need 2 respective functions. So, these are defined and stored in class initializer as primitive components of TrainStep.

#### Backward:
The function `backward()` in PyTorch computes the gradients by backpropagating over computation graph. Here, `backward()` in TrainStep performs forward and backward passes at the same time, also returns the output of both operations. At this point, as you noticed, computing the gradients and using them to update model parameters are separate steps, because python-objects created for neural networks never enclose layer weights unlike PyTorch.

#### Loss Function:
Loss function in TrainState interface is almost same as the one in `train_step()` function except for the parameter *batch*, because its definition is inside class scope, not the scope of `train_step()`. The main functionality of loss function is to perform forward pass and compute final loss. Since normalization layers statistics (mean and variance) are not stored inside the layer just like weights and biases, they need to be returned by the function for updating in `train_step()`. Similarly, to be able to compute performance metrics after one step of training, *logits* are also returned in a pack form together with computed loss and batch stats. 

#### Train Step:
When we look at TrainStep interface closely, we notice that we do not explicitly call `loss_fn()`. Its main role is to specify forward pass operation, and when we pass this loss function to gradient operator in `__init__()`, we actually extend it as a new function where forward and backward passes can be aplicable at the same time. Then, by calling it inside `backward()`, we already perform both operations simultaneously. So, as you see, the functions `backward()` and `loss_fn()` are just primitives. Everything is initiated and finalized inside `train_step()`. To control the training of a neural network, it is sufficient to call `train_step()`; other methods may need to be designed depending on how you conduct your learning model and training stage. For example, if you do not use batch norm layers, the argument `mutable=["batch_stats"]` can be removed from `state.apply_fn()` and `state.replace()` is modified.
