#import everything to use

import os 
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# preparing data

def load_images_from_folder(folder, label, img_size = (64,64)):
    data = []
    for filename in os.listdir(folder): # get picture from folder
        img_path = os.path.join(folder,filename)
        img = cv2.imread(img_path) # read picture 
        if img is not None :
            img = cv2.resize(img,img_size) # change size to the size we settings)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # change to grayscale for highlight structual picture
            data.append((img.flatten() , label)) # change to 1D vector
            
    return data

cat_data = load_images_from_folder(r"D:\Learning_AI_Python\learn_from_video\PetImages\cat_train" , 0 ) # label group cat is 0
dog_data = load_images_from_folder(r"D:\Learning_AI_Python\learn_from_video\PetImages\dog_train" , 1 ) # label group dog is 1

dataset = cat_data + dog_data

x = np.array([features for features , label in dataset])
y = np.array([label for features, label in dataset])

print("Dataset shape:" , x.shape , "Labels:" , y.shape)

# train-test data

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# train classifier

model = LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)

# check acurracy 

ye_pred = model.predict(x_test)
acc = accuracy_score(y_test,ye_pred)
print("Accuracy : " , acc)

# predict image

def predict_image(img_path) :
    img = cv2.imread(img_path)
    img = cv2.resize(img,(64,64))
    img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    img_flat = img.flatten().reshape(1,-1)
    pred = model.predict(img_flat)[0]
    if pred == 1 :
        return "Dog"
    else :
        return "Cat"
print(predict_image(r"D:\Learning_AI_Python\learn_from_video\PetImages\cat_test\12113.jpg"))
print(predict_image(r"D:\Learning_AI_Python\learn_from_video\PetImages\dog_test\12113.jpg"))