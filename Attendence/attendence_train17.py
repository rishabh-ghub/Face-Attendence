import  os
import cv2
import pandas as pd
from PIL import Image
import numpy as np


BASE_DIR=os.path.dirname(os.path.abspath(__file__))
image_dir=os.path.join(BASE_DIR,'images')
face_cascade=cv2.CascadeClassifier(r'C:\Program Files\Python36\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml')
recognizer=cv2.face.LBPHFaceRecognizer_create()

current_id=-1
label_ids=[]   #numbers
y_labels=[]    #names
x_train=[]
label=''
names=[]

for root,dirs,files in os.walk(image_dir):
    for file in files:
        if file.endswith('png') or file.endswith('jpg'):
            path=os.path.join(root,file)
            label=os.path.basename(root).replace(' ','-').lower()
            #print(label,path)
            if not label in y_labels:   
                current_id+=1
            
            #y_labels.append(label)
            #x_train.append(path)
            pil_image=Image.open(path).convert('L')#grayscale
            image_array=np.array(pil_image,'uint8')
            faces=face_cascade.detectMultiScale(image_array)

            for[x,y,w,h] in faces:
                roi=image_array[y:y+h,x:x+w]
                x_train.append(roi)
                y_labels.append(label)
                label_ids.append(current_id)
                
    if(label!=''):
        names.append(label)
        
d=pd.DataFrame({'Names':names})
d.to_csv(r'attendence.csv')
'''print(d)
print(y_labels)
print(label_ids)
print(names)
print(len(x_train))'''


recognizer.train(x_train,np.array(label_ids))
recognizer.save('trainer2.yml')
