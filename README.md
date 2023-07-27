 This Python script that demonstrates how to build, train, and evaluate a Convolutional Neural Network (CNN) model for bone fracture detection using X-ray images. Additionally, it includes a custom callback to stop training when the validation accuracy reaches 80% or higher.

Here's a step-by-step explanation of the code:

Import Libraries: The script starts by importing necessary libraries from TensorFlow and Keras to build and train the CNN model. It also imports other standard libraries like NumPy and os.

Define Dataset Paths and Parameters: The code defines the paths to the training and validation datasets using train_path and val_path. It also specifies the input image size (img_size) and batch size (batch_size) for training.

Data Augmentation: The code sets up two ImageDataGenerator objects, train_datagen and val_datagen, to apply data augmentation and preprocessing to the training and validation datasets. Data augmentation is a technique that creates variations of the images by applying random transformations like shear, zoom, and horizontal flip. This technique helps to increase the diversity of the training data and improves the model's generalization ability.

Load Datasets: The script loads the training and validation datasets using the flow_from_directory method of the ImageDataGenerator. The images are resized to the specified img_size, and the class mode is set to 'binary' since it's a binary classification problem (fracture or non-fracture).

Build the CNN Model: The script defines the architecture of the CNN model using the Sequential class from Keras. The model consists of three sets of convolutional and max-pooling layers to extract features from the input images. After the convolutional layers, the model has a flatten layer to convert the 2D feature maps into a 1D vector, followed by two fully connected (dense) layers with ReLU activation. The output layer uses a sigmoid activation function to perform binary classification.

Compile the Model: The model is compiled using the Adam optimizer with a learning rate of 0.0001, binary cross-entropy loss function (since it's a binary classification problem), and the accuracy metric to monitor the model's performance during training.

Custom Callback: The script defines a custom callback named StopTrainingAtAccuracy. This callback is triggered at the end of each epoch, and it checks the validation accuracy (val_accuracy) in the logs dictionary. If the validation accuracy is 80% or higher, the callback sets self.model.stop_training = True, which stops the training process.

Train the Model: The script trains the model using the fit function. It specifies the number of training epochs (epochs) and uses the data generators train_generator and val_generator to feed the training and validation data to the model. Additionally, it includes the custom callback stop_training_callback in the list of callbacks to monitor the validation accuracy and stop training when the condition is met.

Evaluate the Model: After training, the script evaluates the model's performance on the validation set using the evaluate function and prints the validation loss and accuracy.

Make Predictions: Finally, the script demonstrates how to use the trained model to make predictions on new X-ray images. It loads a new X-ray image, preprocesses it, and then uses the model to predict whether the X-ray image contains a bone fracture or not.

In summary, this code provides a comprehensive example of how to build a CNN model, use data augmentation for training, implement a custom callback, and evaluate the model's performance for bone fracture detection in X-ray images.
