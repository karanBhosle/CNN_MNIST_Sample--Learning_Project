# **CNN_MNIST_Sample - Learning Project**

### **Project Overview:**
This project implements a Convolutional Neural Network (CNN) model for image classification using the popular MNIST dataset. The model classifies images of handwritten digits (0-9) and is built using TensorFlow/Keras. The project includes visualization of the data, training, evaluation, and prediction. The model architecture involves convolutional layers for feature extraction and a dense layer for classification.

### **Key Features:**
- **Data Visualization**: Displays the first 10 images from the MNIST dataset.
- **Model Architecture**: The model consists of convolutional layers, max pooling layers, and dense layers.
- **Prediction and Visualization**: The model can predict the class of a given image and visualize the results.
- **Model Evaluation**: Evaluate the model on test data and calculate the accuracy.
- **Random Image Testing**: Randomly selects test images, compares the predicted and actual class, and calculates accuracy.

### **Technologies Used:**
- **TensorFlow**: For building and training the CNN model.
- **Keras**: For simplifying the neural network layers and model building.
- **NumPy**: For numerical operations.
- **Matplotlib**: For data visualization.
- **MNIST Dataset**: A collection of handwritten digits used for training and testing the model.

### **Getting Started:**
1. **Install Dependencies**: 
   Install the necessary Python packages by running the following command:
   ```bash
   pip install tensorflow numpy matplotlib
   ```

2. **Download Dataset**: 
   The MNIST dataset is automatically loaded from Keras when the script is run.

3. **Run the Code**:
   Execute the provided Python code in a Jupyter notebook or Google Colab environment.

### **Model Architecture:**
The CNN model consists of:
1. **Conv2D Layer (32 filters, 3x3 kernel)**: Extracts features from the input image.
2. **Conv2D Layer (64 filters, 3x3 kernel)**: Further extracts more complex features.
3. **MaxPooling2D Layer (2x2 pool size)**: Reduces the dimensionality of the feature map.
4. **Flatten Layer**: Flattens the 3D data into a 1D vector for feeding into fully connected layers.
5. **Dense Layer (64 neurons)**: A fully connected layer to make decisions based on extracted features.
6. **Dense Layer (10 neurons with softmax activation)**: The output layer that produces a probability distribution for each class (digit 0-9).

### **Training the Model:**
- The model is compiled using the **Adam optimizer** and **categorical cross-entropy loss** function.
- Training is done for **5 epochs** with a **batch size of 128** and **10% validation split**.
  
### **Evaluating the Model:**
- The model is evaluated on the test set, and the **accuracy** is printed.
- A random set of test images are selected, and predictions are made. The accuracy of these random predictions is calculated.

### **Code Structure:**
The code is organized into the following sections:
1. **Import Libraries**: Includes TensorFlow, NumPy, Matplotlib, and Keras.
2. **Data Loading and Preprocessing**: Loads the MNIST dataset and reshapes/normalizes the data.
3. **Model Building**: Defines the architecture of the CNN model.
4. **Model Training**: Trains the model on the training set.
5. **Model Evaluation**: Evaluates the model on the test set and displays the accuracy.
6. **Prediction and Visualization**: Makes predictions on individual images and displays them with the predicted and actual labels.

### **Running the Code:**
- **Step 1**: Import the necessary libraries and load the MNIST dataset.
- **Step 2**: Preprocess the data by reshaping and normalizing the images.
- **Step 3**: Define the CNN architecture and compile the model.
- **Step 4**: Train the model on the training data.
- **Step 5**: Evaluate the model on the test data and visualize predictions.

### **Contact Information:**
- **Project Developed by**: Karan Bhosle
- **LinkedIn Profile**: [Karan Bhosle](https://www.linkedin.com/in/karanbhosle/)
  
Feel free to reach out for questions or collaborations!

