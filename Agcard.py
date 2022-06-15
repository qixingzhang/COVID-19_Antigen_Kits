import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils import *
from PIL import Image, ImageDraw, ImageFont
import shutil
import os

'''
Positive
'''
# img_in = cv2.imread('./positive/p1.png')
# img_in = cv2.imread('./positive/p2.png')
# img_in = cv2.imread('./positive/p3.png')

'''
Negative
'''
img_in = cv2.imread('./negative/IMG_1704.JPG')
# img_in = cv2.imread('./negative/IMG_1705.JPG')

'''
Invalid
'''
# img_in = cv2.imread('./invalid/IMG_1716.JPG')
# img_in = cv2.imread('./invalid/IMG_1717.JPG')
# img_in = cv2.imread('./invalid/IMG_1718.JPG')
# img_in = cv2.imread('./invalid/IMG_1720.JPG')
# img_in = cv2.imread('./invalid/IMG_1731.JPG')
# img_in = cv2.imread('./invalid/IMG_1732.JPG')
# img_in = cv2.imread('./invalid/u1.png')
# img_in = cv2.imread('./invalid/u2.png')
# img_in = cv2.imread('./invalid/u3.png')

'''
Hardmode
'''
# img_in = cv2.imread('./Hardmode/IMG_20220410_130941.jpg')
# img_in = cv2.imread('./Hardmode/IMG_20220411_191854.jpg')
# img_in = cv2.imread('./Hardmode/IMG_20220412_094755.jpg')
# img_in = cv2.imread('./Hardmode/IMG_20220412_172702.jpg')
# img_in = cv2.imread('./Hardmode/IMG_20220413_202942.jpg')
# img_in = cv2.imread('./Hardmode/IMG_20220416_193028.jpg')
# img_in = cv2.imread('./Hardmode/IMG_20220419_135919.jpg')

# img_in = cv2.imread('./Hardmode/IMG_20220421_085725.jpg')
# img_in = cv2.imread('./Hardmode/IMG_20220422_191738.jpg')
# img_in = cv2.imread('./Hardmode/IMG_20220429_193302.jpg')
# img_in = cv2.imread('./Hardmode/IMG_20220430_082437.jpg')
# img_in = cv2.imread('./Hardmode/IMG_20220430_082819.jpg')

img_out, scale, direction, center_all, typeqr_all, box_qrcode, number = find_qrcode(img_in)

img_num_out, num_box_all = find_num_box(img_out, scale, direction, center_all, typeqr_all)
img_antigen_out, antigen_box_all = find_antigen_box(img_num_out, scale, direction, center_all, typeqr_all)
img_fin_out, antigen_all, antigen_center_all = find_antigen(img_antigen_out, scale, direction, center_all, typeqr_all)

# img_antigen_all = extract_antigen(img_fin_out, antigen_box_all, direction, typeqr_all)

def cv2_putText_CN(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontStyle)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

# result_path = './result'
# if not os.path.exists(result_path):
#     os.mkdir(result_path)
# else:
#     shutil.rmtree(result_path)
#     os.mkdir(result_path)

cnt = 0
for i in range(len(antigen_box_all)):
    if typeqr_all[i] >= 0:
        card = split_card(img_in, antigen_all[i], box_qrcode[i], antigen_box_all[i], num_box_all[i], scale[i], direction[i], typeqr_all[i])
        if card != None and card['result'] != 'None':
            img_tmp = card['img']

            img_tmp = cv2.rectangle(img_tmp, card['box_paper'][0], card['box_paper'][1], (0, 0, 255), 2)
            img_tmp = cv2_putText_CN(img_tmp, card['result'], card['box_paper'][0, 0], card['box_paper'][0, 1], (255, 0, 0), 30)

            img_tmp = cv2.rectangle(img_tmp, card['box_SN'][0], card['box_SN'][1], (255, 0, 0), 2)
            SN = card['SN']
            if SN == 'OCR failure':
                if typeqr_all[i] == 1:
                    SN = number[i]
                if typeqr_all[i] == 2 or typeqr_all[i] == 0:
                    SN = '东方基因' + number[i].split('=')[-1]
            img_tmp = cv2_putText_CN(img_tmp, SN, card['box_SN'][0, 0], card['box_SN'][0, 1], (0, 0, 255), 30)

            # print(card['box_qrcode'])
            img_tmp = cv2.rectangle(img_tmp, card['box_qrcode'][0], card['box_qrcode'][1], (0, 255, 0), 2)
            img_tmp = cv2_putText_CN(img_tmp, number[i], card['box_qrcode'][0, 0], card['box_qrcode'][0, 1], (0, 255, 0), 30)
            
            jpg_name = './result_easymode/%s.jpg'%(SN[4:])
            if os.path.exists(jpg_name):
                print('exist', jpg_name)
            print(jpg_name)
            print('antigen center: ', antigen_center_all[i])
            cv2.imwrite(jpg_name, img_tmp)

# plt.figure()
# plt.axis('off')
# plt.imshow(img_fin_out[:, :, [2, 1, 0]])
# plt.show()