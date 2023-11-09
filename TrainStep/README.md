
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
    logits = state.apply_fn({"params": params}, x=x)
    loss = softmax_ce(logits, y)
    return loss, logits

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grads = grad_fn(state.params)
  state = state.apply_gradients(grads)
```

## Questions and Answers:

* *Should we define TrainStep as a function ?*
* TrainStep does not have to be defined as a function. It is also possible to design it as a class like an interface.

$$\$$
* *Why do we define loss function inside `train_step()` ?*
* We define loss function inside `train_step()` so that it can directly access *state* object. Otherwise, `loss_fn()` has to take *state* object as an input argument.
  In this case, we need to specify, by using `argnum=0`, for which input argument (params or state), the gradient of `loss_fn()` will be computed:

```py
grad_fn = jax.value_and_grad(loss_fn, argnum=0, has_aux=True)
```

$$\$$ 
* *Why don't we use `jax.grad()`?*
* We can use `jax.grad()` operator, but `jax.value_and_grad()` performs forward and backward passes at the same time. The latter is more code-efficient.

$$\$$ 
* *Why doesn't loss function just compute loss value, but also perform forward pass ?*
* JAX is capable of computing the gradients of any function, and this gradient computation should not be interrupted. So, forward pass and calculation of loss are put in same
  function.

