import numpy as np
import pandas as pd
import pickle
import input_generation
import os

def targetget(string):

    if 'downstairs' in string:
        targettensor = [0,0,1]
    elif 'normal' in string:
        targettensor = [0,1,0]
    else:
        targettensor = [0,1,0]

    return targettensor

def readhw(sub,HW):

    Height = HW['height [m]']
    Weight = HW['weight [kg]']

    sub_index = HW[HW['subject no.'] == sub].index
    height = Height[sub_index]
    weight = Weight[sub_index]
    heightlist = height.tolist()
    weightlist = weight.tolist()

    if (len(heightlist)==0):
        heightn = 0
        weightn = 0
    else:
        heightn = heightlist[0]
        weightn = weightlist[0]

    return heightn,weightn


path = "D:/Smartphone3/"
HW = pd.read_csv('anthro_2021.csv')

subjects = os.listdir(path)
threshold = 50

input_data = []
target_data = []

for subject in subjects:
    # first read each subject
    files = os.listdir(path + subject)
    subject_number = int(subject[7:10])

    height, weight = readhw(subject_number,HW)
    if 'impaired' in subject:
        continue

    if height==0 or weight==0:
        continue

    heights = np.ones((threshold, 1)) * height
    weights = np.ones((threshold, 1)) * weight
    input1 = np.concatenate([heights, weights], axis=1)

    acceleration = pd.read_csv(path + subject + "/" + "Accelerometer.csv")
    gyroscope = pd.read_csv(path + subject + "/" + "Gyroscope.csv")

    Time = acceleration['Time (s)']
    Xa, Ya, Za = input_generation.readxzya(acceleration, Time)
    Time_G = gyroscope['Time (s)']
    Xg, Yg, Zg = input_generation.readxzyg(gyroscope, Time_G)


    Xa1,Ya1,Za1 = input_generation.dataprocess(Xa,Ya,Za)
    Xg1,Yg1,Zg1 = input_generation.dataprocess(Xg,Yg,Zg)

    if len(Xa1) < threshold or len(Xg1) < threshold:
        continue


    Xa1 = Xa1[:threshold]
    Ya1 = Ya1[:threshold]
    Za1 = Za1[:threshold]
    input_acc = np.stack([Xa1,Ya1,Za1],axis=1)

    Xg1 = Xg1[:threshold]
    Yg1 = Yg1[:threshold]
    Zg1 = Zg1[:threshold]
    input_gyro = np.stack([Xg1,Yg1,Zg1],axis=1)

    input2 = np.concatenate([input_acc,input_gyro],axis=1)
    input = np.concatenate([input1,input2],axis=1)

    target = targetget(subject)

    input_data.append(input)
    target_data.append(target)

input_data = np.array(input_data)
target_data = np.array(target_data)
print(input_data.shape)
print(target_data.shape)

f = open("input.pickle",'wb+')
pickle.dump(input_data,f)
f1 = open("target.pickle",'wb+')
pickle.dump(target_data,f1)


















