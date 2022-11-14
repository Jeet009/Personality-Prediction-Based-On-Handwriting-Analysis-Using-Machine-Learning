import cv2 as cv
import os
import pandas as pd
import numpy as np


def pen_pressure_detection(image):
    list = []
    # Converting the image to greyscale image
    img = cv.imread(image)

    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # COLOR INTENSITY DETECTION

    darker = 0
    lighter = 0
    # traverses through height of the image
    for i in range(gray_image.shape[0]):
        # traverses through width of the image
        for j in range(gray_image.shape[1]):
            if gray_image[i][j] < 100:
                list.append(gray_image[i][j])
            if gray_image[i][j] > 100 and gray_image[i][j] < 200:
                list.append(gray_image[i][j])

    return np.mean(list)
    cv.destroyAllWindows


res = pen_pressure_detection('./srihari.jpeg')
print(res)

# directory = 'data_subset'
# count = 0
# for filename in os.listdir(directory):
#     f = os.path.join(directory, filename)
#     # checking if it is a file
#     if os.path.isfile(f):
#         pen_pressure_detection(f)
#         count = count + 1
#         print(count)
# df = pd.read_excel('./data_new.xlsx')
# df['Pen Pressure'] = pd.Series(np.array(list))
# df.to_excel('data_new.xlsx', index=False)
