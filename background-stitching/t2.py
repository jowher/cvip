# 1. Only add your code inside the function (including newly improted packages). 
#  You can design a new function and call the new function in the given functions. 
# 2. For bonus: Give your own picturs. If you have N pictures, name your pictures such as ["t3_1.png", "t3_2.png", ..., "t3_N.png"], and put them inside the folder "images".
# 3. Not following the project guidelines will result in a 10% reduction in grades

from operator import le
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json


def stitch(imgmark, N=4, savepath=''): #For bonus: change your input(N=*) here as default if the number of your input pictures is not 4.
    "The output image should be saved in the savepath."
    "The intermediate overlap relation should be returned as NxN a one-hot(only contains 0 or 1) array."
    "Do NOT modify the code provided."
    imgpath = [f'./images/{imgmark}_{n}.png' for n in range(1,N+1)]
    imgs = []
    for ipath in imgpath:
        img = cv2.imread(ipath)
        imgs.append(img)
    "Start you code here"
    oneHotEncodingMat = np.zeros((N, N))
    newImgArr = imgs.copy()
    
    for i in range(N):
        for j in range(N):
            if i != j:
                # print("testtt")
                newImgArr[j] = cv2.copyMakeBorder(newImgArr[j], 70, 70, 70, 70, cv2.BORDER_CONSTANT)
                goodPointsLen, warpImg, dst,uncropped = warp(newImgArr[i], newImgArr[j])
                # print("after warp")
                if goodPointsLen < 300:
                    continue
                percentage = calcPercentage(newImgArr[i], dst)
                if percentage > 20:
                    oneHotEncodingMat[i][j] = 1
                    oneHotEncodingMat[j][i] = 1
    for i in range(N):
        for j in range(N):
            if i == j:
                oneHotEncodingMat[i][j] = 1

    # oneHotEncodingMat = [[1, 1, 0, 1], [1, 1, 1, 1], [0, 1, 1, 0], [1, 1, 0, 1]]
    temp = oneHotEncodingMat.copy()
    temp2 = oneHotEncodingMat
    # print(oneHotEncodingMat, "47")
    finalimage = imgs[0]
    for i in range(len(temp)):
        for j in range(len(temp)):
            if i != j:
                if temp[i][j] == 1:
                    finalimage = cv2.copyMakeBorder(finalimage, 70, 70, 70, 70, cv2.BORDER_CONSTANT)
                    goodPointsLen, warpImg, dst,uncropped  = warp(finalimage, imgs[j])
                    if goodPointsLen < 310:
                        goodPointsLen, warpImg, dst,uncropped  = warp( imgs[j],finalimage)
                        if goodPointsLen < 310:
                            continue
                    finalimage = overlap(uncropped, warpImg)
                    temp[j][i] = 0

    cv2.imwrite(savepath, finalimage)
    # print(oneHotEncodingMat)
    overlap_arr = np.asarray(oneHotEncodingMat)
    return overlap_arr

def warp(img1, img2):
    img_gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_gray1,None)
    img_gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    kp2, des2 = sift.detectAndCompute(img_gray2,None)
    des1_len = len(des1)
    goodPoints = []
    # print("warp")
    for i in range(des1_len):
        left = des1[i, :].reshape(1, -1)
        dist = findAndSortDistance(left,des2)
        # print(dist)
        sorted_dist = np.sort(dist)
       
        firstMatch = sorted_dist[0]
        secondMatch = sorted_dist[1]
        first_id= np.where(dist == firstMatch)[0][0]
        if firstMatch < 0.85 * secondMatch: 
            match = {"leftIndex": i, "rightIndex": first_id,  "distance": dist[0]}
            goodPoints.append(match)
    goodPointsLen = len(goodPoints)
    # print(goodPointsLen, "good length")
    source = np.float32([kp1[m.get('leftIndex')].pt for m in goodPoints]).reshape(-1, 1, 2)
    dest = np.float32([kp2[m.get('rightIndex')].pt for m in goodPoints]).reshape(-1, 1, 2)
    # print("float conversion")
    M, mask = cv2.findHomography(dest, source, cv2.RANSAC, 4.0)
    # print(M)
    dst = cv2.warpPerspective(img2, M, ((img2.shape[1] + img1.shape[1]), img1.shape[0]))
    # print("jhjhjjk")
    uncropped_img = dst[0:img1.shape[0], 0:img1.shape[1]]
    warpedImage = dst.copy()
    warpedImage[0:img1.shape[0], 0:img1.shape[1]] = img1
    return goodPointsLen, warpedImage, dst,uncropped_img

def calcPercentage(img1, img2):
    overlapped = 0
    total = 0
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if abs(np.sum(img1[i][j])) == abs(np.sum(img2[i][j])) and abs(np.sum(img1[i][j])) == 0:
                continue
            elif abs(np.sum(img1[i][j])) == abs(np.sum(img2[i][j])):
                overlapped = overlapped + 1
                total = total + 1
            else:
                total = total + 1

    percentage = (overlapped/total)*1000
    return percentage

def overlap(image1, image2):
    output = image2
    row, col = image1.shape[0], image1.shape[1]
    for i in range(row):
        for j in range(col):
            if abs(np.sum(image1[i][j])) == abs(np.sum(image2[i][j])):
                continue
            elif abs(np.sum(image1[i][j])) > abs(np.sum(image2[i][j])):
                output[i][j] = image1[i][j]
            elif abs(np.sum(image1[i][j])) < abs(np.sum(image2[i][j])):
                output[i][j] = image2[i][j]
    return output


def findAndSortDistance(des1, des2):
    return np.sqrt(np.sum(np.square(np.subtract(des1, des2)) , axis=1))


if __name__ == "__main__":
    #task2
    overlap_arr = stitch('t2', N=4, savepath='result.png')
    with open('results.txt', 'w') as outfile:
        json.dump(overlap_arr.tolist(), outfile)



    #bonus
    # overlap_arr2 = stitch('t3', savepath='task3.png')
    # with open('t3_overlap.txt', 'w') as outfile:
    #     json.dump(overlap_arr2.tolist(), outfile)
