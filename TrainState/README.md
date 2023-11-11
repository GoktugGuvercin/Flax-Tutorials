

# TrainState:

* TrainState is a dataclass used to represent the state of model training. In each iteration, the model parameters are optimized, the state of optimizer is updated, and the number of training steps is counted. TrainState is responsible for all these as a representative of training process, but also it can be extended to enclose and manage more data members for state expression of other sub operations in training stage such as calculation of performance metrics and maintenance of batch norm statistics. In that way, it would provide a typical pattern to simplify and modularize the trainin process. When you finish this, you can go to [next tutorial](https://github.com/GoktugGuvercin/Flax-Tutorials/blob/main/TrainStep/README.md)

* Built-in data members to store state information in TrainState are *tx*, *params*, and *step*. They are automatically defined when a a train state object is instantiated. We can subclass TrainState and add new data members like that. The most useful ones are *epoch*, *batch statistics*, and *metrics*:

  |  Built-in | `Definition` |
  | ---       |     ---         |
  | `step`    |  An iteger to count training steps |
  | `params`  | The weights and biases of model  |
  | `tx`      | The optimizer such as Adam or RMSProp  |


  | Extended      | `Definition` |
  | ---           |     ---         |
  | `epoch`       | An iteger to count training epochs |
  | `batch_stats` | The weights and biases of model  |
  | `metrics`     | Eval. metrics like precision and recall |


* When we want to create a TrainState object, we pass necessary arguments for these data members to its class function *create(.)*, which returns a solid TrainState object.
In general, a TrainState object is managed by *train_step(.)* function or a *TrainStep* interface. Which one you want to prefer depends on how you want to design training process. 

* When TrainStep computes the gradients, it calls *apply_gradients(.)* function inside TrainState in order to update the model parameters. This function, at the same time,
  increments step counter by 1 and modify the state of optimizer "tx". However, if we extend TrainState class to add additional data members like *batch_stats* and *epoch*,
  they are not automatically updated by *apply_gradients(.)*; we need to manually modify them with new values by *replace(.)* function.

* Entire TrainState API and its functionalities are provided in the following illustrations. 

# TrainState Layout:

<p align="center">
  <img src="https://github.com/GoktugGuvercin/Flax-Tutorials/blob/main/TrainState/images/TrainState%20Layout.png" width="1000" height="440" />
</p>



# TrainState Functionalities:
 
<p align="center">
  <img src="https://github.com/GoktugGuvercin/Flax-Tutorials/blob/main/TrainState/images/TrainState%20Functionalities.png" width="1000" height="620" />
</p>

