import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, pickle, cv2, glob, gc

DIR_TRAIN = './sign_data/Dataset/train'
DIR_TEST = './sign_data/Dataset/test'
DIR_VAL = './sign_data/Dataset/val'
DATA_TRAIN_CSV = './sign_data/new_train_data.csv'
DATA_TRAIN_ALL_CSV = './sign_data/new_train_all_data.csv'
DATA_TEST_CSV = './sign_data/new_test_data.csv'
DATA_VAL_CSV = './sign_data/new_val_data.csv'

SIZE = 224

# ----------------------------------------------------------
# create train, val and test .csv files with pairs of images
# ----------------------------------------------------------

people_train = dict()
people_train_all = dict()
for person in os.listdir(DIR_TRAIN):
    name = person
    if person[-1]=='g':
        name = person.split('_')[0]
    if name not in people_train_all:
        people_train_all[name] = ([],[])
    if name not in people_train:
        people_train[name] = ([],[])
    for data in glob.glob(DIR_TRAIN + '/' + person + '/*'):
        img = cv2.imread(data)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (SIZE,SIZE))
        if person[-1]=='g':
            people_train_all[name][1].append(data.split("\\")[-1])
            people_train[name][1].append(data.split("\\")[-1])
        else:
            people_train_all[name][0].append(data.split("\\")[-1])
            people_train[name][0].append(data.split("\\")[-1])
c = []
for p in people_train:
    for img1 in people_train[p][0]:
        for img2 in people_train[p][0]:
            if img1 != img2:
                c.append([f'{p}/{img1}', f'{p}/{img2}', 0])
    for img1 in people_train[p][0]:
        for img2 in people_train[p][1]:
            c.append([f'{p}/{img1}', f'{p}_forg/{img2}', 1])

df = pd.DataFrame(c, columns = ['img1','img2','target'])
df.to_csv('./sign_data/new_train_data.csv',index=False) 
            
people_val = dict()
for person in os.listdir(DIR_VAL):
    name = person
    if person[-1]=='g':
        name = person.split('_')[0]
    if name not in people_train_all:
        people_train_all[name] = ([],[])
    if name not in people_val:
        people_val[name] = ([],[])
    for data in glob.glob(DIR_VAL + '/' + person + '/*'):
        img = cv2.imread(data)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (SIZE,SIZE))
        if person[-1]=='g':
            people_train_all[name][1].append(data.split("\\")[-1])
            people_val[name][1].append(data.split("\\")[-1])
        else:
            people_train_all[name][0].append(data.split("\\")[-1])
            people_val[name][0].append(data.split("\\")[-1])

c = []
for p in people_val:
    for img1 in people_val[p][0]:
        for img2 in people_val[p][0]:
            if img1 != img2:
                c.append([f'{p}/{img1}', f'{p}/{img2}', 0])
    for img1 in people_val[p][0]:
        for img2 in people_val[p][1]:
            c.append([f'{p}/{img1}', f'{p}_forg/{img2}', 1])

df = pd.DataFrame(c, columns = ['img1','img2','target'])
df.to_csv('./sign_data/new_val_data.csv',index=False)

c = []
for p in people_train_all:
    for img1 in people_train_all[p][0]:
        for img2 in people_train_all[p][0]:
            if img1 != img2:
                c.append([f'{p}/{img1}', f'{p}/{img2}', 0])
    for img1 in people_train_all[p][0]:
        for img2 in people_train_all[p][1]:
            c.append([f'{p}/{img1}', f'{p}_forg/{img2}', 1])

df = pd.DataFrame(c, columns = ['img1','img2','target'])
df.to_csv('./sign_data/new_train_all_data.csv',index=False) 

people = dict()
for person in os.listdir(DIR_TEST):
    name = person
    if person[-1]=='g':
        name = person.split('_')[0]
    if name not in people:
        people[name] = ([],[])
    for data in glob.glob(DIR_TEST + '/' + person + '/*'):
        img = cv2.imread(data)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (SIZE,SIZE))
        if person[-1]=='g':
            people[name][1].append(data.split("\\")[-1])
        else:
            people[name][0].append(data.split("\\")[-1])
c = []
for p in people:
    for img1 in people[p][0]:
        for img2 in people[p][0]:
            if img1 != img2:
                c.append([f'{p}/{img1}', f'{p}/{img2}', 0])
    for img1 in people[p][0]:
        for img2 in people[p][1]:
            c.append([f'{p}/{img1}', f'{p}_forg/{img2}', 1])

df = pd.DataFrame(c, columns = ['img1','img2','target'])
df.to_csv('./sign_data/new_test_data.csv',index=False) 


# ----------------------
# load and pickle images
# ----------------------

# TRAIN DATA

filenames_train = []  
imgs_train, labels_train = [], []

for person in os.listdir(DIR_TRAIN):
    imgsnames_person = []
    imgsnames_person_forg = []
    for data in glob.glob(DIR_TRAIN + '/' + person + '/*'):
        filenames_train.append(data)
        img = cv2.imread(data)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (SIZE,SIZE))
        imgs_train.append([img])
        if person[-1]=='g':
            labels_train.append(np.array(1))
            imgsnames_person_forg.append(data.split("\\")[-1])  # !!!!! 
        else:
            imgsnames_person.append(data.split("\\")[-1])  # !!!!!
            labels_train.append(np.array(0))    
    
# VALIDATION DATA

filenames_val = []  
imgs_val, labels_val = [], []

for person in os.listdir(DIR_VAL):
    imgsnames_person = []
    imgsnames_person_forg = []
    for data in glob.glob(DIR_VAL + '/' + person + '/*'):
        filenames_val.append(data)
        img = cv2.imread(data)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (SIZE,SIZE))
        imgs_val.append([img])
        if person[-1]=='g':
            labels_val.append(np.array(1))
            imgsnames_person_forg.append(data.split("\\")[-1])  # !!!!! 
        else:
            imgsnames_person.append(data.split("\\")[-1])  # !!!!!
            labels_val.append(np.array(0))

# TRAIN + VALIDATION
imgs_train_all = imgs_train + imgs_val
filenames_train_all = filenames_train + filenames_val
        
# TEST DATA

filenames_test = []
imgs_test, labels_test = [], []

for person in os.listdir(DIR_TEST):
    imgsnames_person = []
    imgsnames_person_forg = []
    for data in glob.glob(DIR_TEST + '/' + person + '/*'):
        filenames_test.append(data)
        img = cv2.imread(data)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (SIZE,SIZE))
        imgs_test.append([img])
        if person[-1]=='g':
            labels_test.append(np.array(1))
            imgsnames_person_forg.append(data.split("\\")[-1]) 
        else:
            imgsnames_person.append(data.split("\\")[-1])
            labels_test.append(np.array(0))      

# labels_train, labels_test : 0/1 je li pravi ili krivotvoreni potpis
labels_train = to_categorical(np.array(labels_train))
labels_val = to_categorical(np.array(labels_val))
labels_test = to_categorical(np.array(labels_test))

imgs_train = np.array(imgs_train)/255.0
imgs_train = imgs_train.reshape(-1, SIZE, SIZE, 3)
imgs_val = np.array(imgs_val)/255.0
imgs_val = imgs_val.reshape(-1, SIZE, SIZE, 3)
imgs_train_all = np.array(imgs_train_all)/255.0
imgs_train_all = imgs_train_all.reshape(-1, SIZE, SIZE, 3)
imgs_test = np.array(imgs_test)/255.0
imgs_test = imgs_test.reshape(-1, SIZE, SIZE, 3)

# save filenames
with open('./train_data_names.pkl', 'wb') as file:
    pickle.dump(filenames_train, file)
with open('./val_data_names.pkl', 'wb') as file:
    pickle.dump(filenames_val, file)
with open('./test_data_names.pkl', 'wb') as file:
    pickle.dump(filenames_test, file)
with open('./train_all_data_names.pkl', 'wb') as file:
    pickle.dump(filenames_train_all, file)
    
# save imgs
with open('./train_imgs.pkl', 'wb') as file:
    pickle.dump(imgs_train, file)
with open('./val_imgs.pkl', 'wb') as file:
    pickle.dump(imgs_val, file)
with open('./test_imgs.pkl', 'wb') as file:
    pickle.dump(imgs_test, file)
with open('./train_all_imgs.pkl', 'wb') as file:
    pickle.dump(imgs_train_all, file)
with open('./train_labels.pkl', 'wb') as file:
    pickle.dump(labels_train, file)
with open('./val_labels.pkl', 'wb') as file:
    pickle.dump(labels_val, file)
with open('./test_labels.pkl', 'wb') as file:
    pickle.dump(labels_test, file)
    