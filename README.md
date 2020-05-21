# Capstone-Project
I built a fake-news and misinformation diagnosis software for my capstone project. The web-based software allow user to input a piece of information or a URL they want to check and will provide a reliability index and s series of references. 
The reliability index is based on four machine learning and deep learning algorithms, the Naive Bayes, the Logistic Regression, the Support Vector Machine, and the Convolutional Neural Network. The list of references is based on a real-time google search. 


## Prerequisites
This software requires the installation of python3. Please get the glove.6B.100d.txt before running the program. 

## Get start
Open the terminal or DOS Shell and go to the program folder. 

Run "python detector.py" to start the server

Open an browser and go to http://localhost:5000 to start the web application.

Select the URL or information and input the content you want to check.

Click the "Check" button and what for the result.

A reliability index together with a list of references will show on the webpage. 

## Code files

train.csv and test.csv are the combined dataset used in my project

DataClean.py: the data preprocessing before running ML models.

search.py: The search program to get URL content and do real-time search to get references

detector.py: The backend server 

index.html: The HTML template for user-interface 

NB.sav, Logistic.sav, SVM.sav, Model_CNN.h5 are the trained models of Naive Bayes, Logistic Regression, SVM, and CNN

NB-LR-SVM.py: The code for developing the NB, LR, SVM

cnn.py: The code for developing CNN

