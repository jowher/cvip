'''
All of your implementation should be in this file.
'''
'''
This is the only .py file you need to submit. 
'''
'''
    Please do not use cv2.imwrite() and cv2.imshow() in this function.
    If you want to show an image for debugging, please use show_image() function in helper.py.
    Please do not save any intermediate files in your final submission.
'''
from helper import show_image

import cv2
import numpy as np
import os
import sys

import face_recognition
from sklearn.cluster import AgglomerativeClustering

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''


def detect_faces(input_path: str) -> dict:
    result_list = []
    # print("detect face")
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    # print(input_path)
    for img in os.listdir(input_path):
        face = cv2.imread(os.path.join(input_path, img))
        scaledFace = cascade.detectMultiScale(face, 1.2, 3)
        # print(scaledFace)
        for x, y, w, h in scaledFace:
            dict={
                    "iname":img,
                     "bbox":[float(x),float(y),float(w+1),float(h-1)]
                }
            # print(dict)
            result_list.append(dict)
    # print(result_list)
    return result_list


'''
K: number of clusters
'''
def cluster_faces(input_path: str, K: int) -> dict:
    result_list = []
    '''
    Your implementation.
    '''
    # print("inside cluster function")
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    faceVectorsArr = []
    faceImgArr = []
    for img in os.listdir(input_path):
        # print(img)
        face = cv2.imread(os.path.join(input_path, img))
        scaledFace = cascade.detectMultiScale(face, 1.2, 3)
        faceImgArr.append(img)
        for x, y, w, h in scaledFace:
            cv2.rectangle(face, (x, y), (x + w, y + h), (0, 0, 0), 2)
            boxes = [(y,x+w,y+h,x)]
            # print(x,y)
            # print(boxes)
            faceVectors = face_recognition.face_encodings(face, boxes)
            # print(faceVectors)
            # faceVectorsArr = np.array(faceVectors)
            faceVectorsArr.append(faceVectors)
    model = AgglomerativeClustering(n_clusters=int(K), affinity='euclidean',  linkage='average')
    faceVectorsArr = np.array(faceVectorsArr)
    newarr = faceVectorsArr.reshape(faceVectorsArr.shape[0], (faceVectorsArr.shape[1]*faceVectorsArr.shape[2]))
    # print(newarr)
    hierarchicalCluster = model.fit(newarr)
    # print(faceVectorsArr.shape)
    # print(hierarchicalCluster.labels_)
    # print(len(hierarchicalCluster.labels_))
    clusterLabels = hierarchicalCluster.labels_
    # clusterObj = {}
    clusterObj = getClusterMapping(clusterLabels, faceImgArr)
    # print("new cluster obj")
    # print(clusterObj)
    for key,val in clusterObj.items():
        new_dict = {
            "cluster_no": int(key),
            "elements": val
        }
        result_list.append(new_dict)
    # print(result_list)
    return result_list


'''
If you want to write your implementation in multiple functions, you can write them here. 
But remember the above 2 functions are the only functions that will be called by FaceCluster.py and FaceDetector.py.
'''

"""
Your implementation of other functions (if needed).
"""


def getClusterMapping(clusterLabels, faceImgArr):
    dict = {}
    for index, ele in enumerate(clusterLabels):
        if ele not in dict:
            dict[ele] = [faceImgArr[index]]
        else:
            dict[ele].append(faceImgArr[index])
    return dict
