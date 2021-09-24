# SKIN CANCER DETECTION APP

## Description
### Skin Cancer Definition and Types: 
- Most skin cancers are locally destructive cancerous (malignant) growth of the skin. They originate from the cells of the epidermis, the superficial layer of the skin and the majority of them rarely spread to other parts of the body and become life-threatening. 
- There are three major types of skin cancer: (1) basal cell carcinoma (the most common), (2) squamous cell carcinoma (the second most common) and (3) melanoma
- The images used for this CNN excercise cover two cancerous types;bcc and mel,together with another 5 benign types; Akiec,Bkl,Df,Nv,and Vasc
| Abbreviation          | Full name                              |
|-----------------------|----------------------------------------|
|Bcc | Basal cell carcinoma |
|Mel| Melanoma |
|Akiec| Actinic keratoses and intraepithelial carcinoma | 
|Bkl | Benign lesions of the keratosis |
|Df | Dermatofibroma |
|Nv | Melanocytic nevi|
|Vasc| Vascular lesions |

## Mission objectives

- Be able to apply a CNN to diagnose the mole images.
- Be able to preprocess data for computer vision and deploy an app that could predict whether the image detected/loaded is cancerous or benign

# Installation

On Windows
virtualenv venv 
\venv\scripts\activate

Or if using linux
python3 -m venv myvenv
source myvenv/bin/activate

Install the requirements:
pip install -r requirements.txt
Run the app:
python app.py

## Python version
* Python 3.8

## Packages used
* os
* pandas
* numpy
* matplotlib.pyplot
* itertools
* seaborn
* PIL
* sklearn
* tensorflow.keras

# Usage
| File                | Description                                                    |
|---------------------|----------------------------------------------------------------|
| mole_detection.py         | Main python code|
| mole_model.py         | Python code with Neural network model|
| mole_preprocessing.py        | Python code for visuals (dataset and results)|
| visuals            | Folder including the plots presented on the Readme |


# The Dataset

The dataset used for the model can be found at  https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000 
It was created by Tschandl et al. 2018. 


# About this app


# Website
https://melanomia-detection.herokuapp.com/

# Contributors
| Name                  | Github                                 |
|-----------------------|----------------------------------------|
|Amaury van Kesteren | https://github.com/AmauryvanKeste | 
|Heba Elabrak | https://github.com/Helabrak |
|Michel OMBESSA | https://github.com/mdifils |
|Minh Hien Vo| https://github.com/minhhienvo368 |

# Timeline
20-09-2021 to 24-09-2021
