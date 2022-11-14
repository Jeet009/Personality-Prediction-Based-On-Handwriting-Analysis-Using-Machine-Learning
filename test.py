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

# DETECTING PEN PRESSURE
# def pen_pressure_detection(image):
#     # Converting the image to greyscale image
#     img = cv.imread(image)

#     gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     # cv.imshow('Grayscale', gray_image)
#     # cv.waitKey(0)

#     # COLOR INTENSITY DETECTION

#     darker = 0
#     lighter = 0
#     # traverses through height of the image
#     for i in range(gray_image.shape[0]):
#         # traverses through width of the image
#         for j in range(gray_image.shape[1]):
#             if gray_image[i][j] < 100:
#                 # print(gray_image[i][j])
#                 darker = darker+1
#             if gray_image[i][j] > 100 and gray_image[i][j] < 200:
#                 # print(gray_image[i][j])
#                 lighter = lighter+1
#     if darker > lighter:
#         print('THE PEN PRESSURE OF TEXT IS HIGH ' + image)
#     else:
#         print('THE PEN PRESSURE OF TEXT IS LOW ' + image)
#     # print(darker)
#     # print(lighter)
#     cv.destroyAllWindows


# EXTRACTING SLANT OF BASELINE
# def slant_of_baseline(image):
#     img = cv.imread(image)

#     gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#     # cv.imshow('Grayscale', gray_image)
#     height_list = []
#     d = pytesseract.image_to_data(
#         gray_image, output_type=pytesseract.Output.DICT)
#     n_boxes = len(d['level'])
#     for i in range(n_boxes):
#         if d['text'][i].strip() == '':
#             continue

#         (x, y, w, h) = (d['left'][i], d['top']
#                         [i], d['width'][i], d['height'][i])
#         height_list.append(d['height'][i])
#         cv.rectangle(gray_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         cv.circle(gray_image, (x, h), 5, (0, 0, 0), -1)

#     print(height_list)
#     # cv.imshow('PLOT', gray_image)
#     if len(height_list) > 0:
#         first_character_height = height_list[0]
#         last_character_height = height_list[len(height_list) - 1]
#         second_last_character_height = height_list[len(height_list) - 2]
#         if (first_character_height - last_character_height > 40 and first_character_height - second_last_character_height > 30):
#             print('BASELINE IS ASSENDING : ' + image)
#         elif (first_character_height - last_character_height < 40 and first_character_height - last_character_height >= -60):
#             print('BASELINE IS STRAIGHT : ' + image)
#         else:
#             print('BASELINE IS DESCENDING : ' + image)
#         # plt.plot(height_list, c="blue")
#         # plt.show()

#         cv.destroyAllWindows


# directory = 'data_subset'
# for filename in os.listdir(directory):
#     f = os.path.join(directory, filename)
#     # checking if it is a file
#     if os.path.isfile(f):
#         slant_of_baseline(f)

# GETTING THE SIZE OF LATER
# from scipy.spatial import distance as dist
# from imutils import perspective
# from imutils import contours
# import numpy as np
# import imutils
# import cv2


# def midpoint(ptA, ptB):
#     return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


# def text_size(img):
#     # load the image, convert it to grayscale, and blur it slightly
#     image = cv2.imread(img)
#     if image is not None:
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         gray = cv2.GaussianBlur(gray, (7, 7), 0)
#         # perform edge detection, then perform a dilation + erosion to
#         # close gaps in between object edges
#         edged = cv2.Canny(gray, 50, 100)
#         edged = cv2.dilate(edged, None, iterations=1)
#         edged = cv2.erode(edged, None, iterations=1)
#         # find contours in the edge map
#         cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
#                                 cv2.CHAIN_APPROX_SIMPLE)
#         cnts = imutils.grab_contours(cnts)
#         # sort the contours from left-to-right and initialize the
#         # 'pixels per metric' calibration variable
#         (cnts, _) = contours.sort_contours(cnts)
#         pixelsPerMetric = None

#         size_dict = []

#         # loop over the contours individually
#         for c in cnts:
#             # if the contour is not sufficiently large, ignore it
#             if cv2.contourArea(c) < 100:
#                 continue
#             # compute the rotated bounding box of the contour
#             orig = image.copy()
#             box = cv2.minAreaRect(c)
#             box = cv2.cv.BoxPoints(
#                 box) if imutils.is_cv2() else cv2.boxPoints(box)
#             box = np.array(box, dtype="int")
#             # order the points in the contour such that they appear
#             # in top-left, top-right, bottom-right, and bottom-left
#             # order, then draw the outline of the rotated bounding
#             # box
#             box = perspective.order_points(box)
#             cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
#             # loop over the original points and draw them
#             for (x, y) in box:
#                 cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

#         # unpack the ordered bounding box, then compute the midpoint
#             # between the top-left and top-right coordinates, followed by
#             # the midpoint between bottom-left and bottom-right coordinates
#             (tl, tr, br, bl) = box
#             (tltrX, tltrY) = midpoint(tl, tr)
#             (blbrX, blbrY) = midpoint(bl, br)
#             # compute the midpoint between the top-left and top-right points,
#             # followed by the midpoint between the top-righ and bottom-right
#             (tlblX, tlblY) = midpoint(tl, bl)
#             (trbrX, trbrY) = midpoint(tr, br)
#             # draw the midpoints on the image
#             cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
#             cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
#             cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
#             cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
#             # draw lines between the midpoints
#             cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
#                      (255, 0, 255), 2)
#             cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
#                      (255, 0, 255), 2)
#             # compute the Euclidean distance between the midpoints
#             dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
#             dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
#             # if the pixels per metric has not been initialized, then
#             # compute it as the ratio of pixels to supplied metric
#             # (in this case, inches)
#             if pixelsPerMetric is None:
#                 pixelsPerMetric = dB / 0.955
#             # compute the size of the object
#             dimA = dA / pixelsPerMetric
#             dimB = dB / pixelsPerMetric
#             # draw the object sizes on the image
#             cv2.putText(orig, "{:.1f}in".format(dimA),
#                         (int(tltrX - 15), int(tltrY - 10)
#                          ), cv2.FONT_HERSHEY_SIMPLEX,
#                         0.65, (0, 0, 0), 2)
#             cv2.putText(orig, "{:.1f}in".format(dimB),
#                         (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
#                         0.65, (0, 0, 0), 2)

#             size_dict.append({'Width': dimA, 'Height': dimB})

#             # _, bt = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
#             # imageout = pytesseract.image_to_string(bt)
#             # print(imageout)

#             # show the output image
#             # cv2.imshow("Image", orig)
#             # cv2.waitKey(0)
#     else:
#         return False

#     print(size_dict, img)


# directory = 'data_subset'
# count = 0
# for filename in os.listdir(directory):
#     f = os.path.join(directory, filename)
#     # checking if it is a file
#     if os.path.isfile(f):
#         res = text_size(f)
#         if res is False:
#             continue
#         count = count + 1
#         print(count)

# def word_spacing(image):
#     img = cv2.imread(image)
#     if img is not None:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         h, w, c = img.shape

#         if w > 1000:

#             new_w = 1000
#             ar = w/h
#             new_h = int(new_w/ar)

#             img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
#         # cv2.imshow(img)

#         def thresholding(image):
#             img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#             ret, thresh = cv2.threshold(
#                 img_gray, 80, 255, cv2.THRESH_BINARY_INV)
#             # cv2.imshow('Thresh', thresh)
#             # cv2.waitKey(0)
#             return thresh

#         thresh_img = thresholding(img)

#         # # dilation
#         # kernel = np.ones((3, 85), np.uint8)
#         # dilated = cv2.dilate(thresh_img, kernel, iterations=1)
#         # plt.imshow(dilated, cmap='gray')

#         # dilation
#         kernel = np.ones((3, 15), np.uint8)
#         dilated2 = cv2.dilate(thresh_img, kernel, iterations=1)
#         # cv2.imshow(dilated2, cmap='gray')

#         (contours, heirarchy) = cv2.findContours(
#             dilated2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#         sorted_contours_lines = sorted(
#             contours, key=lambda ctr: cv2.boundingRect(ctr)[1])  # (x, y, w, h)

#         img3 = img.copy()
#         words_list = []
#         word_distance_list = []
#         prev_end_cord = (0, 0)
#         loop_count = 0
#         height_list = []

#         for line in sorted_contours_lines:

#             # roi of each line
#             x, y, w, h = cv2.boundingRect(line)
#             roi_line = dilated2[y:y+w, x:x+w]

#             # draw contours on each word
#             (cnt, heirarchy) = cv2.findContours(
#                 roi_line.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#             sorted_contour_words = sorted(
#                 cnt, key=lambda cntr: cv2.boundingRect(cntr)[0])

#             for word in sorted_contour_words:

#                 if cv2.contourArea(word) < 400:
#                     continue

#                 x2, y2, w2, h2 = cv2.boundingRect(word)
#                 height_list.append(h2)
#                 words_list.append([(x+x2, y+y2+h2), (x+x2+w2, y+y2+h2)])

#                 cv2.rectangle(img3, (x+x2, y+y2),
#                               (x+x2+w2, y+y2+h2), (255, 255, 100), 2)

#         for word in words_list:
#             loop_count = loop_count + 1
#             im = cv2.circle(img3, word[1], radius=5,
#                             color=(0, 0, 0), thickness=-1)
#             im2 = cv2.circle(img3, word[0], radius=5,
#                              color=(0, 0, 0), thickness=-1)
#             print(math.dist(prev_end_cord, word[0]))
#             word_distance_list.append(math.dist(prev_end_cord, word[0]))
#             prev_end_cord = word[1]

#         # print(word_distance_list)
#         if len(word_distance_list) > 0:
#             print('Avg Distance : ', mean(word_distance_list))
#         # cv2.imshow('image', img3)
#         # cv2.waitKey(0)

#     else:
#         return False


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
