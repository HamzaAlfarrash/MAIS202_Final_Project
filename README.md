# HamdoonTheChatbot
Final project for McGill AI Society Intro to ML Bootcamp (Fall 2022).

#Project Description
HamdoonTheChatbot project is a webapp that hosts Hamdoon, a chatbot trained on 147M Reddit conversations.
The model was built using pytorch and HugginFace's transformers, and the web app's backend
was built using flask.

#Running the app
to run the web app, install all packages in requirements.txt. then, change into the main
directory of this repository and run

python3 app.py

Lastly, open a browser and navigate to your http://localhost:5000.

#Repository organization
This repository contains the scripts used to both train the model and build the web app.

1. The .pdf files
   - deliverables submitted to the MAIS Intro to ML Bootcamp organizers
2. templates/
   - HTML template for landing page
3. app.py
   -main python script to instantiate Flask server and train the model
