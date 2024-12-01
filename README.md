Guava Disease Classifier
This project is a deep learning-based classifier designed to detect diseases in guava fruits. 
It classifies guavas into three categories: Anthracnose, Fruit Flies, and Healthy fruits. 
The project includes a trained model and a Streamlit app to interact with the classifier.

Project Structure
dataprocess.py: Preprocessing and data handling script for loading and preparing images.
guava_disease_classifier.py: The Streamlit app to interact with the model and visualize predictions.
model_checkpoints: Checkpoint files for the trained deep learning model.
model_checkpoint_*.ckpt.index: Index files for each checkpoint of the model training process.

Dataset
The dataset used for training is not included in this repository due to its size:
You can download it here: https://www.kaggle.com/datasets/asadullahgalib/guava-disease-dataset
The dataset consists of 3,784 augmented images of guava fruits, categorized into three classes:
Anthracnose (a fungal disease),
Fruit Flies (insect infestation),
Healthy fruits.
The images were collected from guava orchards in Bangladesh and preprocessed for deep learning tasks.

Model
The classifier is built using TensorFlow/Keras and trained on the dataset. 
The model architecture uses CNN layers to learn features from the images and predict the health status of guavas.
The model was saved using TensorFlow checkpoints for large-scale data handling.

How to Run
Clone the repository:
git clone https://github.com/ashpunia/guava-disease-classifier.git

Install dependencies:
pip install -r requirements.txt

Run the Streamlit app:
streamlit run guava_disease_classifier.py

Upload guava images through the Streamlit interface to get predictions.

Checkpoints and Model Files
The trained model's weights are saved as TensorFlow checkpoints. 
You can find the checkpoint index files (.ckpt.index) in the repository. 
These checkpoints store information about the model's layers and weights, which can be loaded for further training or inference.
