from asyncore import loop
from ctypes.wintypes import WORD
from itertools import count
from statistics import mean
from telnetlib import WONT
import cv2
import os
import pytesseract
from PIL import Image
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

list = []


def word_spacing(image):
    img = cv2.imread(image)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, c = img.shape

        if w > 1000:

            new_w = 1000
            ar = w/h
            new_h = int(new_w/ar)

            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # cv2.imshow(img)

        def thresholding(image):
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(
                img_gray, 80, 255, cv2.THRESH_BINARY_INV)
            # cv2.imshow('Thresh', thresh)
            # cv2.waitKey(0)
            return thresh

        thresh_img = thresholding(img)

        # # dilation
        # kernel = np.ones((3, 85), np.uint8)
        # dilated = cv2.dilate(thresh_img, kernel, iterations=1)
        # plt.imshow(dilated, cmap='gray')

        # dilation
        kernel = np.ones((3, 15), np.uint8)
        dilated2 = cv2.dilate(thresh_img, kernel, iterations=1)
        # cv2.imshow(dilated2, cmap='gray')

        (contours, heirarchy) = cv2.findContours(
            dilated2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        sorted_contours_lines = sorted(
            contours, key=lambda ctr: cv2.boundingRect(ctr)[1])  # (x, y, w, h)

        img3 = img.copy()
        words_list = []
        word_distance_list = []
        prev_end_cord = (0, 0)
        loop_count = 0
        height_list = []

        for line in sorted_contours_lines:

            # roi of each line
            x, y, w, h = cv2.boundingRect(line)
            roi_line = dilated2[y:y+w, x:x+w]

            # draw contours on each word
            (cnt, heirarchy) = cv2.findContours(
                roi_line.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            sorted_contour_words = sorted(
                cnt, key=lambda cntr: cv2.boundingRect(cntr)[0])

            for word in sorted_contour_words:

                if cv2.contourArea(word) < 400:
                    continue

                x2, y2, w2, h2 = cv2.boundingRect(word)
                height_list.append(h2)
                words_list.append([(x+x2, y+y2+h2), (x+x2+w2, y+y2+h2)])

                cv2.rectangle(img3, (x+x2, y+y2),
                              (x+x2+w2, y+y2+h2), (255, 255, 100), 2)

        for word in words_list:
            loop_count = loop_count + 1
            im = cv2.circle(img3, word[1], radius=5,
                            color=(0, 0, 0), thickness=-1)
            im2 = cv2.circle(img3, word[0], radius=5,
                             color=(0, 0, 0), thickness=-1)
            # print(math.dist(prev_end_cord, word[0]))
            word_distance_list.append(math.dist(prev_end_cord, word[0]))
            prev_end_cord = word[1]

        # print(word_distance_list)
        if len(word_distance_list) > 0:
            # print('Avg Distance : ', mean(word_distance_list))
            return mean(word_distance_list)
        # cv2.imshow('image', img3)
        # cv2.waitKey(0)

    else:
        return False


res = word_spacing('./srihari.jpeg')
print(res)


# directory = 'data_subset'
# count = 0
# for filename in os.listdir(directory):
#     f = os.path.join(directory, filename)
#     # checking if it is a file
#     if os.path.isfile(f):
#         res = word_spacing(f)
#         if res is False:
#             continue
#         count = count + 1
#         print(count)
# df = pd.DataFrame(list, columns=['Word Spacing'])
# print(df)
# data_set = {'Data Set 1': df}
# writer = pd.ExcelWriter('./data.xlsx', engine='xlsxwriter')
# data_set['Data Set 1'].to_excel(
#     writer, sheet_name='Mini Project', index=False)
# writer.save()
