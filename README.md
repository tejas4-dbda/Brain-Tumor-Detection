<h1>Brain Tumor Classification using Deep Learning</h1>

Developed a deep learning model to classify brain tumors from MRI images into four categories: Glioma, Meningioma, Pituitary, and No Tumor. Leveraged the VGG16 architecture for transfer learning, fine-tuning the model to achieve high accuracy in detecting and classifying brain tumors. Implemented data fetching from MongoDB to train the model, enhancing its performance and reliability.

<h2>Table of Contents</h2>

•	Installation

•	Dataset

•	Data Preprocessing

•	Model Architecture

•	Training and Validation

•	Evaluation

•	Results

•	Usage

•	Acknowledgements

<h2>Installation</h2>
To set up the environment and run the code, follow these steps:
<h3>1.	Clone the repository:</h3>
bash

git clone https://github.com/tejas4-dbda/brain-tumor-classification.git

cd brain-tumor-classification

<h3>2.	Install the required packages:</h3>

bash

pip install -r requirements.txt


<h2>Dataset</h2>
The dataset is structured as follows:

•	dataset3/All_images: Contains all images of brain tumors.

•	datasetno/no_tumor: Contains images with no tumor.


The images are categorized into four classes:

•	Glioma

•	Meningioma

•	Pituitary

•	No Tumor

The dataset is split into training, validation, and testing sets.
<h2>Data Preprocessing</h2>

The preprocessing pipeline includes:

<h4>1.	Reading and Loading Data:</h4> Images are loaded from the directories, resized, and stored into arrays.

<h4>2.	Normalization:</h4> Images are normalized to a range of 0-255.

<h4>3.	Cropping:</h4> Images are cropped to focus on the area of interest using contour detection.

<h4>4.	Data Augmentation:</h4> Augmented with transformations such as rotation, flipping, and brightness adjustments.

<h4>5.	Preprocessing for VGG16:</h4> Images are preprocessed to match the input requirements of the VGG16 model.


<h2>Model Architecture</h2>

![Architecture.png](https://raw.githubusercontent.com/Geetanshu18/Brain-Tumor-Detection/main/Architecture.png)

The model is based on the VGG16 architecture, a pre-trained convolutional neural network (CNN) used for image classification tasks. The architecture has been fine-tuned for the specific task of brain tumor classification.

Key components:

<h4>•	Convolutional Layers:</h4> Extract features from the images.

<h4>•	Fully Connected Layers:</h4> Classify the features into one of the four classes.

<h4>•	Softmax Activation:</h4> Used in the output layer for multi-class classification.

<h2>Training and Validation</h2>

The training and validation process involves:

<h4>•	Data Generators:</h4> Use ImageDataGenerator for on-the-fly data augmentation.

<h4>•	Training:</h4> The model is trained using the augmented data.

<h4>•	Validation:</h4> Performance is evaluated on a separate validation set to prevent overfitting.

</h3>Callbacks</h3>

The code uses an early stopping mechanism to halt training if the validation accuracy does not improve.

<h2>Evaluation</h2>

The model is evaluated using various metrics such as accuracy, confusion matrix, and more. Visualization of the distribution of the different classes across the datasets (training, validation, testing) is done using Plotly.

<h2>Results</h2>
The results include:
<h4>•	Accuracy:</h4> Achieved accuracy on the test set.

<h4>•	Confusion Matrix:</h4> To visualize the model's performance in classifying the four types of images.

<h4>•	Sample Plots:</h4> Visualizations of the images and their corresponding predictions.

<h2>Usage</h2>

<h3>Running the Code</h3>

1.	<h4>Upload Images to MongoDB:</h4> If required, you can upload the images to MongoDB using the provided scripts.

2.	<h4>Preprocess and Train the Model:</h4>

    o	Preprocess images and organize them into directories for training, validation, and testing.

    o	Train the model using the training set, and validate it using the validation set.

3.	<h4>Evaluate the Model:</h4>

    o	After training, evaluate the model using the test set to see how well it generalizes to unseen data.

4.	<h4>Visualize Results:</h4>

    o	Visualize the classification results using the provided functions.

<h3>Example Commands</h3>
<h4>•	Load and Preprocess Data:<h4>

python

X_train, y_train, labels = load_data(TRAIN_DIR, IMG_SIZE)

<h4>•	Train the Model:</h4>

python

model.fit(train_generator, validation_data=validation_generator, epochs=50)

<h4>•	Evaluate the Model:</h4>

python

model.evaluate(X_test_prep, y_test)

<h2>Acknowledgements</h2>
This project is based on the analysis and classification of brain tumor images using deep learning techniques. Special thanks to the open-source community for providing the tools and libraries used in this project.

