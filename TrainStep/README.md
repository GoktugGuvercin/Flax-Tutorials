
# TrainStep:

TrainStep, with the simplest terms, is actually a function to carry out one step of model training stage. In other words, it performs forward pass to compute logits and loss, and 
then perform backpropagation for gradient calculation, and update all model parameters with those gradients. By doing that, it completes all necessary operations to train the 
network one step. To be able to do these easily, it utilizes *grad()* or *value_and_grad()* JAX operator and *apply_gradients()* function inside TrainState object. You can find a 
simple code example below.

```py
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
