<div align="center">
    <img width="250" height="250" src="./assets/Cyclonic_Logo.png">
</div>

# Cyclonic AI Module
This module is part of the Cyclonic Project for my MTU Final Year Project.

This module is responsible for the intialisation and running of previously trained AI models, based on the data fetched from the API using Cyclonic's Data Module.

## Installation

### Prerequisites
Ensure that you have Anaconda installed.
Ensure that you have a machine with a DEDICATED GPU or else you will not be able to run this project.

### Installation Steps
Clone the master branch of this repository and go into the directory:

```bash
git clone https://github.com/supraaxdd/cyclonic-ai.git
cd cyclonic-ai
```

To run this project effectively, you should create an anaconda environment using the provided environment.yml file

```bash
conda env create --name cyclonic-env --file=environment.yml
```

and then activate the environment using:

```bash
conda activate cyclonic-env
```

## Training the AI model

Step 1: To train the model, you need to first fetch the data using the [Cyclonic Data module](https://github.com/supraaxdd/cyclonic-data). Refer to the documentation on its Github page for steps on how to fetch the data.

Step 2: Once the data has been retrieved, you need to create a folder called `input` in the cyclonic-ai project directory.

Step 3: In this directory, you need to copy and paste the result.json from the `output` folder of the data module into this folder.

Step 4: Now, you run the Long-Short Term Memory model using:

```bash
python .\lstm.py
```

This will start training the model. 


## Running a prediction on a trained model.

Once a model is trained, as outlined in the [Training the AI model](##training-the-ai-model), you can run it using the `run_saved_model.py` script. To predict the future wind speeds, you need to fetch future data using the [Cyclonic Data module](https://github.com/supraaxdd/cyclonic-data). Refer to the documentation on its Github page for steps on how to fetch the data.

Do steps 1 - 3 in the [Training the AI model](##training-the-ai-model) section to get the future data. 

After completing those steps, run the script by doing:

```bash
python .\run_saved_model.py
```

This will run the model that was saved from training in the `saved` folder. 