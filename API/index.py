import json
import cv2 as cv
import cv2
import math
import pandas as pd
import numpy as np
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import os
import pandas as pd
from statistics import mean
import pytesseract
from flask import Flask, jsonify, request
from flask_restful import Resource, Api
import pickle

app = Flask(__name__)
api = Api(app)


# Feature Extraction
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def pen_pressure_detection(image):
    list = []
    img = cv.imread(image)

    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    for i in range(gray_image.shape[0]):
        for j in range(gray_image.shape[1]):
            if gray_image[i][j] < 100:
                list.append(gray_image[i][j])
            if gray_image[i][j] > 100 and gray_image[i][j] < 200:
                list.append(gray_image[i][j])

    return np.mean(list)


def text_size(img):
    image = cv2.imread(img)
    if image is not None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        edged = cv2.Canny(gray, 50, 100)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        (cnts, _) = contours.sort_contours(cnts)
        pixelsPerMetric = None

        size_dict = []

        for c in cnts:
            if cv2.contourArea(c) < 100:
                continue

            orig = image.copy()
            box = cv2.minAreaRect(c)
            box = cv2.cv.BoxPoints(
                box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")

            box = perspective.order_points(box)
            cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

            for (x, y) in box:
                cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)

            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)

            cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

            cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                     (255, 0, 255), 2)
            cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                     (255, 0, 255), 2)

            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            if pixelsPerMetric is None:
                pixelsPerMetric = dB / 0.955

            dimA = dA / pixelsPerMetric
            dimB = dB / pixelsPerMetric

            cv2.putText(orig, "{:.1f}in".format(dimA),
                        (int(tltrX - 15), int(tltrY - 10)
                         ), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 0, 0), 2)
            cv2.putText(orig, "{:.1f}in".format(dimB),
                        (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 0, 0), 2)

            size_dict.append({'Width': dimA, 'Height': dimB})

    else:
        return False

    length = len(size_dict)
    width_list = []
    height_list = []

    for i in size_dict:
        width_list.append(i['Width'])
        height_list.append(i['Height'])
    if len(width_list) != 0:
        if mean(height_list) > (1.2402 + .25):
            return 2
            #         print('BIG')
        elif mean(height_list) < (1.2402 - .25):
            return 0
            #         print('SMALL')
        else:
            return 1
            #         print('NORMAL')


def slant_of_baseline(image):
    img = cv.imread(image)

    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    height_list = []
    d = pytesseract.image_to_data(
        gray_image, output_type=pytesseract.Output.DICT)
    n_boxes = len(d['level'])
    for i in range(n_boxes):
        if d['text'][i].strip() == '':
            continue

        (x, y, w, h) = (d['left'][i], d['top']
                        [i], d['width'][i], d['height'][i])
        height_list.append(d['height'][i])
        cv.rectangle(gray_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv.circle(gray_image, (x, h), 5, (0, 0, 0), -1)

    if len(height_list) > 0:
        first_character_height = height_list[0]
        last_character_height = height_list[len(height_list) - 1]
        second_last_character_height = height_list[len(height_list) - 2]
        if (first_character_height - last_character_height > 40 and first_character_height - second_last_character_height > 30):
            return 2
            # list.append('ASSENDING')
        elif (first_character_height - last_character_height < 40 and first_character_height - last_character_height >= -60):
            return 1
            # list.append('STRAIGHT')
        else:
            return 0
            # list.append('DESCENDING')


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

        def thresholding(image):
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(
                img_gray, 80, 255, cv2.THRESH_BINARY_INV)
            return thresh

        thresh_img = thresholding(img)

        kernel = np.ones((3, 15), np.uint8)
        dilated2 = cv2.dilate(thresh_img, kernel, iterations=1)

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

            x, y, w, h = cv2.boundingRect(line)
            roi_line = dilated2[y:y+w, x:x+w]

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
            word_distance_list.append(math.dist(prev_end_cord, word[0]))
            prev_end_cord = word[1]

        if len(word_distance_list) > 0:
            return mean(word_distance_list)

    else:
        return False


res_pen_pressure = pen_pressure_detection('./image.jpeg')
res_text_size = text_size('./image.jpg')
res_slant_of_baseline = slant_of_baseline('./image.jpeg')
res_word_spacing = word_spacing('./image.jpeg')
print(res_word_spacing)

penPressure = round(res_pen_pressure)
baselineStraight = 0
wordSizeNormal = 0
wordSizeSmall = 0
wordSpacing = round(res_word_spacing)


if res_slant_of_baseline == 1:
    baselineStraight = 1
else:
    baselineStraight = 0


if res_text_size == 1:
    wordSizeNormal = 1
elif res_text_size == 2:
    wordSizeNormal = 0
    wordSizeSmall = 0
else:
    wordSizeSmall = 1


# Prediction With ML Model

with open('../model_pickle_emo', 'rb') as f:
    model_emo = pickle.load(f)

with open('../model_pickle_att', 'rb') as f:
    model_att = pickle.load(f)

with open('../model_pickle_opt', 'rb') as f:
    model_opt = pickle.load(f)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class Prediction(Resource):

    def get(self):
        res_emo = model_emo.predict(
            [[penPressure, baselineStraight, wordSizeNormal, wordSizeSmall, wordSpacing]])[0]
        res_att = model_att.predict([[177, 1, 0, 1, 425]])[0]
        res_opt = model_opt.predict([[177, 1, 0, 1, 425]])[0]
        # print(res)
        # return jsonify({'square': res**2})
        return json.dumps({'Emotion': res_emo, 'Attention': res_att, 'Optimism': res_opt}, cls=NpEncoder)

    def post(self):
        # print(request)
        # data = request.get_json()
        file = request.files['image']
        file.save('image.jpg')

        res_pen_pressure = pen_pressure_detection('./image.jpg')
        res_text_size = text_size('./image.jpg')
        res_slant_of_baseline = slant_of_baseline('./image.jpg')
        res_word_spacing = word_spacing('./image.jpg')
        print(res_word_spacing)

        penPressure = round(res_pen_pressure)
        baselineStraight = 0
        wordSizeNormal = 0
        wordSizeSmall = 0
        wordSpacing = round(res_word_spacing)

        if res_slant_of_baseline == 1:
            baselineStraight = 1
        else:
            baselineStraight = 0

        if res_text_size == 1:
            wordSizeNormal = 1
        elif res_text_size == 2:
            wordSizeNormal = 0
            wordSizeSmall = 0
        else:
            wordSizeSmall = 1

        res_emo = model_emo.predict(
            [[penPressure, baselineStraight, wordSizeNormal, wordSizeSmall, wordSpacing]])[0]
        res_att = model_att.predict(
            [[penPressure, baselineStraight, wordSizeNormal, wordSizeSmall, wordSpacing]])[0]
        res_opt = model_opt.predict(
            [[penPressure, baselineStraight, wordSizeNormal, wordSizeSmall, wordSpacing]])[0]
        return json.dumps({'Emotion': res_emo, 'Attention': res_att, 'Optimism': res_opt}, cls=NpEncoder)


api.add_resource(Prediction, '/')


# driver function
if __name__ == '__main__':

    app.run(debug=True)
