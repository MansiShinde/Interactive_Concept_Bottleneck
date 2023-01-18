# Interactive_Concept_Bottleneck

Caltech-UCSD Birds 200 2011 (CUB-200-2011) Dataset is used as the training and test data. Please download the dataset via https://data.caltech.edu/records/65de6-vp158

Below is the folder structure of the application:
- BOTTLENECK_UI – main folder
- backend
-models - models saved for implementing independent bottleneck pipeline
first_model.pt - Neural network model
second_model.sav - multiclass classifier model
- static
styles.css - css file to style the UI
- concept_bottleneck.py - the main flask server file
- configs.py - constants like batch_size, no of classes used to train model declared here
- data_preprocess.py - data preprocessing steps mentioned
- dataset.py - creating dataset class and loading data
- load_model.py - loads the uploaded models
- metrics.py - calculating accuracies and binary accuracies
- predict_rerun.py - inference script for prediction and rerun
- test_pipeline.py - code to test pipeline developed using independent bottleneck
- train_independent.py - contains steps done during training the first & second model
- train_sequential.py - contains steps done during training via sequential approach
- CUB_200_2011 - dataset used (this is not present in the source code submitted. Mentioned here for reference to know the location of the dataset used in the source code)
- templates
- BottleNeckUI.html - html file for UI and connecting with server file.


Steps to run the application:

Following are the steps of installation:
1. Create a Conda environment using the command below. Check the latest python version on your system and accordingly put the python version in the command:
conda create -n env_pytorch python=3.9.12
2. Activate the environment using: conda activate env_pytorch
3. Now install PyTorch using pip: pip3 install torchvision
4. Install flask pip3 install flask
5. pip3 install scikit-learn scipy matplotlib numpy


After completing above steps, go to BOTTLENECK_UI/backend folder and run the below command to start the application:

-> cd BOTTLENECK_UI/backend

-> python concept_bottleneck.py

It will get the below result :
* Serving Flask app 'concept_bottleneck'
* Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
* Running on http://127.0.0.1:5000
Press CTRL+C to quit
The application will be running on http://127.0.0.1:5000 server.



https://user-images.githubusercontent.com/29672533/213049573-78fc53b4-837a-4290-81c5-0d8ef31369d3.mov



