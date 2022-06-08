import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils import *

img_in = cv2.imread('./Hardmode/IMG_20220416_193028.jpg')

img_out, scale, direction, center_all, typeqr_all = find_qrcode(img_in)
img_num_out, num_box_all = find_num_box(img_out, scale, direction, center_all, typeqr_all)
img_antigen_out, antigen_box_all = find_antigen_box(img_num_out, scale, direction, center_all, typeqr_all)
for i in range(len(scale)):
    result = pred_antigen(img_in, scale[i], direction[i], center_all[i], typeqr_all[i])
    cv2.putText(img_antigen_out, result, antigen_box_all[i][3, :], cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 3)
    print(result)

plt.figure()
plt.imshow(img_antigen_out[:, :, [2, 1, 0]])
plt.show()
