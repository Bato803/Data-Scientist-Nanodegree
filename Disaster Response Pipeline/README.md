# Disaster Response Pipeline Project

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Introduction
This is a really cool project, it's about how we should response to a disaster efficiently by utilizing machine learning model. Typically following a disaster, we got millions and millions of messages posting on the web via social media. Meanwhile, disaster response organization needs to filter out all those messages and pull out the messages which are the most important. Normally, only one in a thousand messages that might be relevant to the response prefessionals. And we also need to pull out different categories from the data, so that organizations with different focus could easily found out what's most relevant to them. Some of themmight be helping water supply, some for medical equippments, some for blocked roads.
 
Our job, is to define a machine learning problem, to classify thousands of messages into different categories, and found out those that disaster reponse teams are most interested in.

## Files:
- data/process\_data.py: Clean and process dataset into a format that's friendly to Machine Learning pipeline.  
- models/train\_classifier.py: Train a machine learning algorithm to learn from the data output by 'process\_data.py', evaluate and output the model into pickle file. 
- run.py: The file users need to run in order to start the web app.  

## Examples:

<img src="https://github.com/Bato803/Data-Scientist-Nanodegree/blob/master/Disaster%20Response%20Pipeline/images/sample01.png" width = 650, height=300>

<img src="https://github.com/Bato803/Data-Scientist-Nanodegree/blob/master/Disaster%20Response%20Pipeline/images/sample02.png" width=650, height=300>

<img src="https://github.com/Bato803/Data-Scientist-Nanodegree/blob/master/Disaster%20Response%20Pipeline/images/graph.png" width=650, height=300>
