<h1>Brain Tumor Classification using Deep Learning</h1>

Developed a deep learning model to classify brain tumors from MRI images into four categories: Glioma, Meningioma, Pituitary, and No Tumor. Leveraged the VGG16 architecture for transfer learning, fine-tuning the model to achieve high accuracy in detecting and classifying brain tumors. Implemented data fetching from MongoDB to train the model, enhancing its performance and reliability.



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

•	dataset3/All_images: Contains all images of brain tumors.(https://www.kaggle.com/datasets/adityakomawar/dataset3)

•	datasetno/no_tumor: Contains images with no tumor.(https://www.kaggle.com/datasets/adityakomawar/datasetno)


The images are categorized into four classes:

•	Glioma

•	Meningioma

•	Pituitary

•	No Tumor



<h2>Model Architecture</h2>

![Architecture.png](https://raw.githubusercontent.com/Geetanshu18/Brain-Tumor-Detection/main/Architecture.png)

The model is based on the VGG16 architecture, a pre-trained convolutional neural network (CNN) used for image classification tasks. The architecture has been fine-tuned for the specific task of brain tumor classification.

Key components:

<h4>•	Convolutional Layers:</h4> Extract features from the images.

<h4>•	Fully Connected Layers:</h4> Classify the features into one of the four classes.

<h4>•	Softmax Activation:</h4> Used in the output layer for multi-class classification.


