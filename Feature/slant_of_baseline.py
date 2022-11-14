import cv2 as cv
import os
import pytesseract
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

list = []


def slant_of_baseline(image):
    img = cv.imread(image)

    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # cv.imshow('Grayscale', gray_image)
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

    print(height_list)
    # cv.imshow('PLOT', gray_image)
    if len(height_list) > 0:
        first_character_height = height_list[0]
        last_character_height = height_list[len(height_list) - 1]
        second_last_character_height = height_list[len(height_list) - 2]
        if (first_character_height - last_character_height > 40 and first_character_height - second_last_character_height > 30):
            return 2
            # list.append('ASSENDING')
        elif (first_character_height - last_character_height < 40 and first_character_height - last_character_height >= -60):
            return 0
            # list.append('STRAIGHT')
        else:
            return 1
            # list.append('DESCENDING')
        # plt.plot(height_list, c="blue")
        # plt.show()

        cv.destroyAllWindows


res = slant_of_baseline('./srihari.jpeg')
print(res)

# count = 0
# directory = 'data_subset'
# for filename in os.listdir(directory):
#     f = os.path.join(directory, filename)
#     # checking if it is a file
#     if os.path.isfile(f):
#         slant_of_baseline(f)
#         count = count + 1
#         print(count)

# df = pd.read_excel('./data.xlsx')
# df['Baseline'] = pd.Series(np.array(list))
# df.to_excel('data.xlsx', index=False)
