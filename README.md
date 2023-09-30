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


	
### Training Models	

#### Training and evaluating the model	
Tell Keras how many training passes we want it to do over the training data during the training process. 	
A single training pass across the training data set is called an epoch.

If we do too few passes, the neural network wont make accurate predictions, but if we do too many it will waste time, and it might also cause over fitting problems.

The best way to tune this is to try training the neural network and stop doing additional training passes when the network stops getting more accurate.
	
Shuffle the training data randomly. Neural network typicall train best when the data is shuffled. So we'll pass in shuffle equals to true.

Verbose=2 - this simply tells Keras to print more detailed information during training so we can watch what's going on.
```
model.fit(X, Y, epochs=50, shuffle=True, verbose=2)
```

To measure the error rate of the testing data, we'll model.evaluate
```
test_error_rate = model.evaluate(X_test, Y_test, verbose=0)	
```

#### Making predictions

```	
# Make a prediction with the neural network
prediction = model.predict(X)

# Grab just the first element of the first prediction (since that's the only have one)
prediction = prediction[0][0]

# Re-scale the data from the 0-to-1 range back to dollars
# These constants are from when the data was originally scaled down to the 0-to-1 range
prediction = prediction + 0.1159
prediction = prediction / 0.0000036968
```	



#### Saving and loading models	
To save the Keras model, we call model.save	and pass the file name. 
\
When we save the model, it save both the structure of the neural network and the trained weights that determine how the neural network works.
\
The reason we use the h5 extension is because data will be stored in the HDF Five format. \
HDF Five format is a binary file format designed for storing Python array data. \
The convention is to use h5 as the filename extension but it's not required. 

```
# Save the model to disk
model.save("trained_model.h5")
print("Model saved to disk.")
```

```
from keras.models import load_model

model = load_model("trained_model.h5")
```

### Pre-Trained Models in Keras
#### Pre-trained models


#### Recognized images with RestNet50 model
ImageNet
A dataset of millions of labelled pictures
Used to train image recognition models

ILSVR - ImageNet Large Scale Visual Recognition Challenge
Yearly image recognition competition

There are four types of pre-trained image recognition models included with Keras
* VGG (Visual Geometry Group at University of Oxford) - VGG is a Deep Neural network, with 16 or 19 layers. State of art from 2014 and still widely used today, but and takes a lot of memory to run.
* ResNet50 (Microsoft Research) - State of the from 2015. Its a 50-layer neural network that manages to be more accurate with less memory but still use less memory that the VGG design.
* Inception-v3 (Google) - is another design from 2015 that also performs very well.
* Xception (Francois Chollet, author or Keras) - Xception, is an improve version of Incention-v3. More accurate than v3 while using the same amount of memory.

```
import numpy as np
from keras.preprocessing import image
from keras.applications import resnet50

# Load Keras' ResNet50 model that was pre-trained against the ImageNet database
model = resnet50.ResNet50()

# Load the image file, resizing it to 224x224 pixels (required by this model)
img = image.load_img("bay.jpg", target_size=(224, 224))

# Convert the image to a numpy array
x = image.img_to_array(img)

# Add a forth dimension since Keras expects a list of images
x = np.expand_dims(x, axis=0)

# Scale the input image to the range used in the trained network
x = resnet50.preprocess_input(x)

# Run the image through the deep neural network to make a prediction
predictions = model.predict(x)

# Look up the names of the predicted classes. Index zero is the results for the first image.
predicted_classes = resnet50.decode_predictions(predictions, top=9)

print("This is an image of:")

for imagenet_id, name, likelihood in predicted_classes[0]:
    print(" - {}: {:2f} likelihood".format(name, likelihood))

```

### Monitoring a Keras model with TensorBoard

#### Export Keras logs in TensorFlow format
```
# Define the model
model = Sequential()
model.add(Dense(50, input_dim=9, activation='relu', name='layer_1'))
model.add(Dense(100, activation='relu', name='layer_2'))
model.add(Dense(50, activation='relu', name='layer_3'))
model.add(Dense(1, activation='linear', name='output_layer'))
model.compile(loss='mean_squared_error', optimizer='adam')

# Create a TensorBoard logger
logger = keras.callbacks.TensorBoard(log_dir="logs", write_graph=True, histogram_freq=5)

# Train the model
model.fit(
    X,
    Y,
    epochs=50,
    shuffle=True,
    verbose=2,
    callbacks=[logger]
)
```

#### Visualize the computational graph
```
tensorboard --loadir=06\logs
```

#### Visualize training progress

```
RUN_NAME = "run 1 with 50 nodes"

# Create a TensorBoard logger
logger = keras.callbacks.TensorBoard(
    log_dir='logs {}'.format(RUN_NAME),
    histogram_freq=5,
    write_graph=True
)
```	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
