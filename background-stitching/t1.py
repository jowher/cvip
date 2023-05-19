#Only add your code inside the function (including newly improted packages)
# You can design a new function and call the new function in the given functions. 
# Not following the project guidelines will result in a 10% reduction in grades

from dis import dis
import cv2
import numpy as np
import matplotlib.pyplot as plt


def stitch_background(img1, img2, savepath=''):
    "The output image should be saved in the savepath."
    "Do NOT modify the code provided."
    
    img1 = cv2.copyMakeBorder(img1, 300, 300, 300, 300, cv2.BORDER_CONSTANT)
    #cv2.imwrite('warped2.jpg', img1)

    img_gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_gray1,None)
    img_gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    kp2, des2 = sift.detectAndCompute(img_gray2,None)
    # print(des2)
    des1_len = len(des1)
    # des2_len = len(des2)

    # Finding the matching points in both images by calculating the ssd distance.
    goodPoints = []
    for i in range(des1_len):
        left = des1[i, :].reshape(1, -1)
        dist = findAndSortDistance(left,des2)
        sorted_dist = np.sort(dist)
        firstMatch = sorted_dist[0]
        first_id= np.where(dist == firstMatch)[0][0]
        if dist[0] < 0.8 * dist[1]: # Comparing first and second match
            match = {"leftIndex": i, "rightIndex": first_id,  "distance": dist[0]}
            goodPoints.append(match)
    # print(len(goodPoints))
    source = np.float32([kp1[m.get('leftIndex')].pt for m in goodPoints]).reshape(-1, 1, 2)
    dest = np.float32([kp2[m.get('rightIndex')].pt for m in goodPoints]).reshape(-1, 1, 2)
    # print(source)
    M, mask = cv2.findHomography(dest, source, cv2.RANSAC, 5.0)
    # print(M)
    dst = cv2.warpPerspective(img2, M, ((img2.shape[1] + img1.shape[1]), img1.shape[0]))
    uncropped_img = dst[0:img1.shape[0], 0:img1.shape[1]]

    image_final = overlap(uncropped_img, img1)
    # image_final_brdr = cv2.copyMakeBorder(image_final, 100, 100, 100, 100, cv2.BORDER_CONSTANT, None, value = 0)
    # image_final_brdr = cv2.copyMakeBorder(image_final, 100, 100, 100, 100, cv2.BORDER_CONSTANT, None, value = 0)
    cv2.imwrite(savepath, image_final)
    return

#Function to overlap image and remove foreground.
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



#Calculating the distance
def findAndSortDistance(des1, des2):
    return np.sqrt(np.sum(np.square(np.subtract(des1, des2)) , axis=1))

if __name__ == "__main__":
    img1 = cv2.imread('./images/t1_1.png')
    img2 = cv2.imread('./images/t1_2.png')
    savepath = 'task1.png'
    stitch_background(img1, img2, savepath=savepath)

