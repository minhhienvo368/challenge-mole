# SKIN CANCER DETECTION USING CNN AND KERAS PROJECT

## Description
### Skin Cancer Definition and Types: 
- Most skin cancers are locally destructive cancerous (malignant) growth of the skin. They originate from the cells of the epidermis, the superficial layer of the skin and the majority of them rarely spread to other parts of the body and become life-threatening, ecxept for Melanoma which is lethal.
- There are three major types of skin cancer: (1) basal cell carcinoma (the most common), (2) squamous cell carcinoma (the second most common) and (3) melanoma
- The images used for this CNN excercise cover two cancerous types; bcc and mel,together with another 5 benign types; Akiec,Bkl,Df,Nv,and Vasc

     | Abbreviation          | Full name                              |
     |-----------------------|----------------------------------------|
     |Bcc | Basal cell carcinoma |
     |Mel| Melanoma |
     |Akiec| Actinic keratoses and intraepithelial carcinoma | 
     |Bkl | Benign lesions of the keratosis |
     |Df | Dermatofibroma |
     |Nv | Melanocytic nevi|
     |Vasc| Vascular lesions |
 

### Project mission:
- The main mission of this project was to create an app using CNN and Keras to diagnose the mole images. The app could predict whether the image detected/loaded is cancerous or benign


## NocCan App 

### Website
https://melanomia-detection.herokuapp.com/

## Visuals:  
![Snapshot of the App](https://user-images.githubusercontent.com/84380899/134656153-df62d463-dcf0-4d06-a6bc-c3ac3ea3d78c.png)


## Installation on local machine

**On Windows*
$ virtualenv venv 
$ \venv\scripts\activate

**Or if using Linux/ MACOS**
$ python3 -m venv myvenv
$ source myvenv/bin/activate

**Install the requirements:**
$ pip install -r requirements.txt

**Run the app:**
$ python app.py

### Python version
* Python 3.9

### Packages used
* os
* numpy==1.19.5
* pandas==1.3.3
* matplotlib.pyplot==3.4.2
* itertools
* seaborn
* sklearn
* tensorflow.keras
* Keras==2.4.3
* Pillow==8.3.1
* scikit-learn==0.24.2
* streamlit==0.88.0
* tensorflow-cpu==2.5.0

## Usage and links

   | File                | Description                                                    |
   |---------------------|----------------------------------------------------------------|
   | mole_detection.py         | Main python code|
   | mole_model.py         | Python code with Neural network model|
   | mole_preprocessing.py        | Python code for visuals (dataset and results)|
   | visuals            | Folder including the plots presented on the Readme |https://github.com/mdifils/cancer-detector
   | NoCan App           | https://github.com/mdifils/cancer-detector|
   | NoCan App Presentation           | Powerpoint presentation 

## The Dataset

The dataset used for the model can be found at  https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000 
It was created by Tschandl et al. 2018. 



## Contributors
   | Name                  | Github                                 |
   |-----------------------|----------------------------------------|
   |Amaury van Kesteren | https://github.com/AmauryvanKeste | 
   |Heba Elabrak | https://github.com/Helabrak |
   |Michel OMBESSA | https://github.com/mdifils |
   |Minh Hien Vo| https://github.com/minhhienvo368 |

## Timeline
20-09-2021 to 24-09-2021
