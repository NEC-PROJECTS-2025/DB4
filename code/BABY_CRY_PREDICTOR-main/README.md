Baby Cry Detection
This repository contains an end-to-end project for detecting the reasons behind a baby's cry
using machine learning techniques and deploying the model using Flask. Get dataset from kaggle
Overview
The project involves training a machine learning model to classify the reasons behind a
baby's cry into five classes: tired or lack of sleep, burping, hunger or exhaustion, discomfort
or lack of aff ection and attention, and belly pain or colic. The trained model is then deployed
using Flask to create a web application where users can upload audio recordings of a baby's
cry and receive predictions on the reason behind the cry.
Project Structure
Usage
1. Training the Model: Run the `baby_cry_detection.ipynb` notebook to train the machine
learning model on the dataset provided in the `donateacry` folder.
2. Deploying the Model: Use the `app.py` file to deploy the trained model using Flask. Ensure
that all necessary dependencies are installed.
3. Running the Web Application: Access the web application by running the Flask server and
navigating to the provided URL. Users can upload audio recordings and receive predictions
on the reason behind the cry.
Additional Notes
Dependencies
baby_cry_detection.ipynb: Jupyter Notebook containing the code for training the 
machine learning model.
 app.py: Flask application for deploying the trained model.
 templates: Folder containing HTML templates for the web application.
 index.html: HTML file for the home page where users can upload audio recordings.
 result.html: HTML file for displaying the prediction results.
uploads: Folder where uploaded audio files are stored temporarily.
 donateacry: Folder containing the dataset used for training the model.
cried: Additional folder for storing audio files for testing purposes.
 For testing purposes, additional audio files can be added to the `cried` folder.
The accuracy of the model on each class is provided in the `baby_cry_detection.ipynb` 
notebook.
Feel free to modify the HTML templates or Flask routes to customize the web application
according to your preferences.
Python 3.x
Flask
NumPy
Librosa
Pickle
