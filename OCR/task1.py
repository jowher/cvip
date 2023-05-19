"""
Character Detection

The goal of this task is to implement an optical character recognition system consisting of Enrollment, Detection and Recognition sub tasks

Please complete all the functions that are labelled with '# TODO'. When implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them.

Do NOT modify the code provided.
Please follow the guidelines mentioned in the project1.pdf
Do NOT import any library (function, module, etc.).
"""


import argparse
import json
from multiprocessing.dummy import Array
from operator import ne
import os
import glob
import cv2 
import numpy as np
import csv

def read_image(img_path, show=False):
    """Reads an image into memory as a grayscale array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if show:
        show_image(img)

    return img

def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--test_img", type=str, default="./data/test_img.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--character_folder_path", type=str, default="./data/characters",
        help="path to the characters folder")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args

def ocr(test_img, characters):
    """Step 1 : Enroll a set of characters. Also, you may store features in an intermediate file.
       Step 2 : Use connected component labeling to detect various characters in an test_img.
       Step 3 : Taking each of the character detected from previous step,
         and your features for each of the enrolled characters, you are required to a recognition or matching.

    Args:
        test_img : image that contains character to be detected.
        characters_list: list of characters along with name for each character.

    Returns:
    a nested list, where each element is a dictionary with {"bbox" : (x(int), y (int), w (int), h (int)), "name" : (string)},
        x: row that the character appears (starts from 0).
        y: column that the character appears (starts from 0).
        w: width of the detected character.
        h: height of the detected character.
        name: name of character provided or "UNKNOWN".
        Note : the order of detected characters should follow english text reading pattern, i.e.,
            list should start from top left, then move from left to right. After finishing the first line, go to the next line and continue.
        
    """

    enrollment(characters)

    componentMap = detection(test_img)
    print("component map")
    # print(componentMap)
    result = recognition(test_img,componentMap)
    return result

def enrollment(characters):
    """ Args:
    You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    featureMap = {}
    # gray= cv2.cvtColor(characters[1],cv.COLOR_BGR2GRAY)
    # show_image(characters[1][1])

    for i in range(0, len(characters)):
        gray = characters[i][1]
        # show_image(gray,2000)
        img = cv2.resize(gray, (250, 250))
        img = np.pad(img, [(2,), (2,)], mode='constant', constant_values=(255))
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(img, None)

        # img=cv2.drawKeypoints(gray,kp,img)
        # show_image(img, 2000)
        # print(des)
        # print(i)
        if (des is not None):
            featureMap[characters[i][0]] = des.tolist()
    with open("features2.json", "w") as outputFile:
        json.dump(featureMap, outputFile)


def detection(test_img):
    """
    Use connected component labeling to detect various characters in an test_img.
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    if test_img is None:
        print("here")
        return {}
    testImageM = test_img.shape[0]
    testImageN = test_img.shape[1]

    threshold = 180
    newArray = np.zeros((testImageM, testImageN))
    for i in range(testImageM):
        for j in range(testImageN):
            if test_img[i][j] < threshold:
                newArray[i][j] = 1
            else:
                newArray[i][j] = 0
    # show_image(newArray)
    # print(newArray)
    # show_image(newArray)
    # newArray=np.array([[1,0,0,0,0],
    # [0,0,1,1,0],
    # [1,1,1,1,0],
    # [0,0,0,0,0],
    # [0,0,0,0,1]])

    level = 2
    labelMap = {}
    # print(newArray)
    x = []
    for i in range(len(newArray)):
        for j in range(len(newArray[i])):
            if newArray[i][j] == 1:
                # labelMap[level]= []
                x = bfs(newArray, i, j, level)
                if (x[2] > 1 and x[3] > 1):
                    labelMap[level] = x
                level += 1
    return labelMap


def bfs(image, i, j, seq):
    queue = []
    queue.append([i, j])
    dir = [[1, 0], [0, 1], [1, 1], [0, -1], [-1, 0], [-1, -1], [1, -1], [-1, 1]]
    x = [float('inf'), float('inf'), float('-inf'), float('-inf')]
    while (len(queue) != 0):
        size = len(queue)
        for i in range(size):
            temp = queue.pop()
            for num in range(len(dir)):
                row = dir[num][0] + temp[0]
                col = dir[num][1] + temp[1]
                if (row > 0 and col > 0 and row < len(image) and col < len(image[0])):
                    if (image[row][col] != 0 and image[row][col] != seq):
                        image[row][col] = seq
                        queue.append([row, col])
                        x[0] = min(x[0], col)
                        x[1] = min(x[1], row)
                        x[2] = max(x[2], col)
                        x[3] = max(x[3], row)

    height = x[3] - x[1]
    width = x[2] - x[0]
    return [x[0], x[1], width, height]


def recognition(img, componentMap):
    """
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    outputResult = []
    newDict = {}
    sift = cv2.SIFT_create()
    with open('features2.json') as inputFile:
        features = json.load(inputFile)
    for label in features:
        if len(features[label]) == 0:
            continue
        des = np.asarray(features[label])
        if des is None:
            continue
        currMin = float('inf')
        for key, value in componentMap.items():
            imageBbox = img[value[1]:value[1] + value[3], value[0]:value[0] + value[2]]
            imageBbox = np.pad(imageBbox, [(4,), (4,)], mode='constant', constant_values=(255))
            imageBbox = cv2.resize(imageBbox, (250, 250))
            kp, des2 = sift.detectAndCompute(imageBbox, None)
            if des2 is None:
                continue
            for i in range(len(des2)):
                for j in range(len(des)):
                    ssd = np.sqrt(np.sum(np.square(np.subtract(des2[i], des[j]))))
                    currMin = min(currMin, ssd)
                    newDict[label] = currMin
    for key, value in componentMap.items():
        imageBbox = img[value[1]:value[1] + value[3], value[0]:value[0] + value[2]]
        imageBbox = np.pad(imageBbox, [(5,), (5,)], mode='constant', constant_values=(255))
        imageBbox = cv2.resize(imageBbox, (250, 250))
        # show_image(imageBbox,1000)

        kp, des2 = sift.detectAndCompute(imageBbox, None)
        tempDict = {}
        tempDict["bbox"] = value
        tempDict["name"] = "Unknown"
        if des2 is None:
            # outputResult.append(tempDict)
            continue
        flag = False
        for labelNew, desNew in features.items():
            if len(desNew) == 0:
                continue
            desNew = np.asarray(desNew)
            count = 0
            # print(label,len(des))
            for i in range(len(des2)):
                for j in range(len(desNew)):
                    ssd = np.sqrt(np.sum(np.square(np.subtract(des2[i], desNew[j]))))
                    if ssd < newDict[labelNew] * 2:
                        count += 1
                        if count >= 2:
                            flag = True
                            tempDict["name"] = labelNew
                            break
                if flag:
                    break
        if flag:
            outputResult.append(tempDict)
        else:
            outputResult.append({
                "bbox": value,
                "name": "UNKNOWN"
            })
    return outputResult


def calc_norm(x1, x2):
# #   return np.sum(np.square(np.subtract(x1, x2)))
    return np.sqrt(np.sum((x1 - x2) ** 2))



   
                    
# def calc_norm(x1, x2): 
# #   return np.sum(np.square(np.subtract(x1, x2)))
#   return np.sqrt(np.sum((x1 - x2) ** 2))

def save_results(coordinates, rs_directory):
    """
    Donot modify this code
    """
    results = coordinates
    with open(os.path.join(rs_directory, 'results2.json'), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()
    
    characters = []

    all_character_imgs = glob.glob(args.character_folder_path+ "/*")
    
    for each_character in all_character_imgs :
        character_name = "{}".format(os.path.split(each_character)[-1].split('.')[0])
        characters.append([character_name, read_image(each_character, show=False)])

    test_img = read_image(args.test_img)

    results = ocr(test_img, characters)

    save_results(results, args.rs_directory)


if __name__ == "__main__":
    main()
