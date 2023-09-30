# Deep Learning
**Keras** - a high level framework for building neural network
Keras build the backend using TensorFlow or Theano
Keras is a front-end layer
**Theano** - Created at MILA (Montreal Institute for Learning Algorithms) at the University of Montreal. \
**TensorFlow** - Created at Google.

**TensorFlow Alone** - low level, more control, write more code
* Researching new types of machine learning models
* Building a large-scale system to support many users
* processing and memory efficiency
	
**Keras + TensorFlow** - High levelm Fast experimentation, less code
* Education and experimentation
* Prototyping

## Creating a Neural Network in Keras
### The train-test-evaluation flow
#### Creating Neural Network in Keras
**Supervised Learning** - the process to follow called the model, train test evaluation low. \
Step 1: Choose Model \
Step 2: Training Phase \
Step 3: Testing Phase - load second set of data that never seen by the model. \
Step 4: Evaluation Phase

#### Create a Model object
```
# the model object represents the neural network we are building
model = keras.models.Sequential() 

# we can add layers to NN just by calling model.add and passing the type of layers we want to add
models.add(keras.layers.Dense()) 
#.... add more layers ....

# the final steps in defining the model is to compile it.
# thats when Keras actually builds a tensor flow model
# how to measure accuracy (loss function), which optimizer algorithms
model.compile(loss='mean_squared_error', optimizer='adam')
	
# training the data.
model.fit(training_ata, expected_output)
	
# test phase
error_rate = model.evaluate(testing_data, expected_output)
	
# if we are happy of the accuracy
model.save("trained_model.h5")
	
# evaluation phase.
model = keras.models.load_model('trained_model.h5")
	
predictions = model.predict(new_data)
```


#### Keras Sequential API

A neural network is a ML algorith that made up individual nodes called neurons
This nodes / neuros are arrange in groups called layers
	
**When designing the NN:**
* How many layers should be?
* How many nodes should be in each layer
* How the layers are connected to each other


**Keras Sequential Model API:**
Easiest way to build a NN in Keras
	
Its called Sequential Model because you create an empty model object and then you add layers to it one after another in sequence.

```
model = keras.models.Sequential()

# We are adding Densely connected layer of 32 node to the NN. A densely connected layer is one where every mode,
# input_dim - need to define for the very first layer. 
model.add(Dense(32, input_dim=9) 
model.add(Dense(128))
model.add(Dense(1))
```

**Customizing Layers**
Before values flow from nodes in one layer to the next, they pass through an activation function \
Keras lets us choose which activation function is used for each layer by passing in the name of the activation function \
relu - rectified linear unit 
 
```
model.add(Dense(number_of_neurons, activation="relu"))
```
_(The default settings are good start)_
	
#### Other Types of Layers Supported
**Convolutional layers**
Typically used to process images and special data
Example: 
```
keras.layers.convolutional.Conv2D()
```
	
**Recurrent layers**
Special layers that have a memory built into each neuron.
Previous data points are important understanding the next data point.
Example: 
``` 
keras.layers.recurrent.LSTM()
```

builds the model defined in TensorFlow backend.
optimizer algo is the algo used to train your neural network
loss function measures how right or how wrong your NN predictions

``` model.compile(optimizer='adam', loss='mse') ```
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
