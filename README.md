# Disaster Response Pipeline Project
# Table of Contents
1. Introduction
2. File Structure
3. Instruction
4. Additional Material
5. Results
6. Acknowledgements

# Introduction
This Project is a part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight. The initial dataset contains pre-labelled tweet and messages from real-life disaster. The aim of the project is to build a Natural Language Processing (NLP) tool that categorize disaster messages.
The Project is divided in the following three sections:<br>
1. <b>ETL Pipeline</b> : extract data from source, clean data and save them in a proper database structure<br>
2. <b>ML Pipeline</b> : train a model able to classify text messages into appropriate categories<br>
3. <b>Flask Web App</b> : show model results in real time<br>

# File Structure
- <b>App</b> folder contains the following:<br>
  &emsp;<b>templates</b>: folder containing<br>
  &emsp;&emsp;<b>go.html</b>: renders the message claddifier<br>
  &emsp;&emsp;<b>master.html</b>: renders homepage<br>
  &emsp;<b>run.py</b>: defines the app routes<br>

- Data folder contains the following:<br>
  <b>categories.csv</b>: contains the disaster categories csv file<br>
  <b>messages.csv</b>: contains the disaster messages csv file<br>
  <b>DisasterResponse.db</b>: contains the emergency db which is a merge of categories and messages by ID<br>
  <b>process_data.py</b>: contains the scripts to transform data<br>

- Model folder contains the following:<br>
  <b>classifier.pkl</b>: contains the RandomForestClassifier pickle file<br>
  <b>train_classifier.py</b>: script to train ml_ipeline.py<br>

# Instruction
To run the pipelines:<br>
- Run the ETL pipeline via the command line in the 'data' folder:<br>
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db<br>
- Run the ML pipeline via the command line in the 'model' folder:<br>
python train_classifier.py ../data/DisasterResponse.db disaster_model.pkl<br>

To run the web app:<br>
- Execute the python file 'run.py' in the 'app' folder via the command line:<br>
python run.py<br>
- Go to http://0.0.0.0:3001/<br>

# Additional Material
I upload the following jupyter notebooks to the 'Data' and 'Model' folder, so that it will help you understand how the model works step by step:<br>
- ETL Pipeline Preparation Notebook: an implemented ETL pipeline which extracts, transforms, and loads raw dataset into a cleaned dataset.<br>
- ML Pipeline Preparation Notebook: analyzing machine learning models through NLP process to find the final model.<br>

# Results
1. After you open the link, you should be able to see the web app. After entering message and clicking <b>Classify Message</b>, you can see the categories which the message belongs to<br>
![image](https://github.com/YaqingChen123/Udacity-Project-DisasterResponse/blob/main/Image/Disaster%20Response%20Project1.png)
2. The following screenshot is the result<br>
![image](https://github.com/YaqingChen123/Udacity-Project-DisasterResponse/blob/main/Image/Disaster%20Response%20Project2.png)
3. The web app's main page shows an overview of traning data
![image](https://github.com/YaqingChen123/Udacity-Project-DisasterResponse/blob/main/Image/Disaster%20Response%20Project3.png)

# Acknowledgements
I'm grateful to <b>Udacity</b> for enabling me to apply my data engineering skills to build a model for an API that classifies disaster messgae. Also thanks to <b>Figure Eight</b> for providing message dataset to train my model
