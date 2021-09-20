import os
import pandas as pd

df_meta = pd.read_csv("Data/HAM10000_metadata.csv")
print(df_meta)
df_meta.describe()
df_meta.info()

base_dir = 'Data'
image_class = df_meta['dx'].unique().tolist()

def cancer_or_not(dx_value: str) -> str:
    if dx_value in ['bcc', 'mel']:
        output = 'cancer'
    else:
        output = 'not_cancer'
    return output

df_meta['labels'] = df_meta['dx'].apply(cancer_or_not)
print(df_meta.head())

y = df_meta['dx']
X = df_meta.drop('dx', axis=1)

#split data
from sklearn.model_selection import train_test_split

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, 
    y,
    test_size=0.3, 
    random_state=42, 
    shuffle=True
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, 
    y_train_val,
    test_size=0.2, 
    random_state=42, 
    shuffle=True
)

# Transfer the images into folders, Set the image id as the index
df_meta.set_index('image_id', inplace=True)

# Get a list of images in each of the two folders
folder_1 = os.listdir('Data/HAM10000_images_part_1')
folder_2 = os.listdir('Data/HAM10000_images_part_2')

# Get a list of train and val images
train_list = list(X_train['image_id'])
val_list = list(X_val['image_id'])
test_list = list(X_test['image_id'])