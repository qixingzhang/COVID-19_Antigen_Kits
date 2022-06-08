import numpy as np
import cv2
import math
from pyzbar.pyzbar import decode
import scipy.signal as signal

def rotate(img, angle):
    '''
    img   --image
    angle --rotation angle
    return--rotated img
    '''
    h, w = img.shape[:2]
    rotate_center = (w/2, h/2)
    M = cv2.getRotationMatrix2D(rotate_center, angle, 1.0)
    new_w = int(h * np.abs(M[0, 1]) + w * np.abs(M[0, 0]))
    new_h = int(h * np.abs(M[0, 0]) + w * np.abs(M[0, 1]))
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    rotated_img = cv2.warpAffine(img, M, (new_w, new_h))
    return rotated_img

def qrcode(img):
    code = decode(img)
    degree = 0
    out = 0
    if len(code) == 0:
        rotate(img,90)
        degree = 90
        code = decode(img)
        if len(code) == 0:
            rotate(img,90)
            degree = 180
            code = decode(img)
            if len(code) == 0:
                rotate(img,90)
                degree = 270
                code = decode(img)
                if len(code) == 0:
                    out = 2021231057
    typeqr = 0        
    for code_inside in code:
        out = code_inside.data.decode("utf-8")
        if len(out) == 0:
            print("no valid number found")
        else:
            if out[0] == 'h':
                typeqr = 0
            elif out[0] == 'A':
                typeqr = 1
            elif out[0] == 'O':
                typeqr = 2
            else:
                typeqr = 3
    return out, typeqr, degree

def findtype(number, rotateimg, typeqr):
    num, typeqrt ,degree = qrcode(rotateimg)
    if num == 2021231057:
        typeqrt = 3
    else:
        number.append(num)
    return typeqrt

def finddirection(bin, number, rotateimgb, box, typeqr):
    
    # orient gene
    if typeqr == 0:
        a0 = np.around(0.865 * box[0] + 0.135 * box[2])
        a1 = np.around(0.865 * box[1] + 0.135 * box[3])
        a2 = np.around(0.865 * box[2] + 0.135 * box[0])
        a3 = np.around(0.865 * box[3] + 0.135 * box[1])
    # ANBF
    else:
        a0 = np.around(0.833 * box[0] + 0.167 * box[2])
        a1 = np.around(0.833 * box[1] + 0.167 * box[3])
        a2 = np.around(0.833 * box[2] + 0.167 * box[0])
        a3 = np.around(0.833 * box[3] + 0.167 * box[1])

    s0 = 0
    s1 = 0
    s2 = 0
    s3 = 0
    
    for ii in range(11):
        for jj in range(11):
            s0 = s0 + bin[int(a0[1]-5+ii)][int(a0[0]-5+jj)]
            s1 = s1 + bin[int(a1[1]-5+ii)][int(a1[0]-5+jj)]
            s2 = s2 + bin[int(a2[1]-5+ii)][int(a2[0]-5+jj)]
            s3 = s3 + bin[int(a3[1]-5+ii)][int(a3[0]-5+jj)]
                  
    sm = np.max([s0,s1,s2,s3])                 
           
    r=1
    while sm == 0:
        for ii in range(11+r):
            for jj in range(11+r):
                s0 = s0 + bin[int(a0[1]-5+ii)][int(a0[0]-5+jj)]
                s1 = s1 + bin[int(a1[1]-5+ii)][int(a1[0]-5+jj)]
                s2 = s2 + bin[int(a2[1]-5+ii)][int(a2[0]-5+jj)]
                s3 = s3 + bin[int(a3[1]-5+ii)][int(a3[0]-5+jj)]
        sm = np.max([s0,s1,s2,s3]) 
        r = r+1
    
    if sm == s0:
        return 0
    elif sm == s1:
        return 270
    elif sm == s2:
        return 180
    elif sm == s3:
        return 90

def find_qrcode(bgr_img):
    color = bgr_img.copy()
    img = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

    _, img = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)

    img = cv2.Canny(img, 100, 200)

    kernel = np.ones((5, 5), np.uint8)
    img = cv2.dilate(img, kernel, iterations=5)
    img = cv2.erode(img, kernel, iterations=4)

    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    squares = []
    areas = []
    rects = []

    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        rect = cv2.minAreaRect(contours[i])

        h = rect[1][0]
        w = rect[1][1]

        if (abs(w - h) < h*0.15) and area < 200000:
            areas.append(area)
            squares.append(contours[i])
            rects.append(rect)
    max_area = max(areas)

    scale = []
    angle = []
    direction = []
    number = []
    center_all = []

    for i in range(len(squares)):
        if areas[i] > max_area*0.4 and areas[i] < max_area*2:
            box = np.int0(cv2.boxPoints(rects[i]))
            area_new = cv2.contourArea(box)
            if abs(area_new - areas[i]) < areas[i]*0.9:

                M = cv2.moments(box)
                center = np.zeros(2)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                center[0] = cX
                center[1] = cY
                center = center.astype(int)
                center_all.append(center)

                cv2.drawContours(color, [box], -1, (255, 0, 0), 5)
                
                scale.append(np.sqrt(area_new))
                angle.append(90-rects[i][2])
                
                rotateimg = np.zeros((2*int(np.around(np.sqrt(area_new))),2*int(np.around(np.sqrt(area_new)))))
                
                cen = np.around(0.5 * box[0] + 0.5 * box[2])

                for ii in range(2*int(np.around(np.sqrt(area_new)))):
                    for jj in range(2*int(np.around(np.sqrt(area_new)))):
                        rotateimg[ii][jj] = gray[int(cen[1])-int(np.around(np.sqrt(area_new)))+ii][int(cen[0])-int(np.around(np.sqrt(area_new)))+jj]
                
                _, rotateimgb = cv2.threshold(rotateimg, 70, 255, cv2.THRESH_BINARY)

                rotateimgb = rotate(rotateimgb, rects[i][2] - 90)
                typeqr = 1

                typeqr = findtype(number, rotateimgb, typeqr)
                
                thres = 70

                while typeqr == 3:
                    print("binary adjust applied")
                    if thres < 170:
                        thres = thres + 10
                        typeqr = 1
                        _, rotateimgb = cv2.threshold(rotateimg, thres, 255, cv2.THRESH_BINARY)
                        rotateimgb = rotate(rotateimgb,rects[i][2] - 90)
                        typeqr = findtype(number,rotateimgb,typeqr)
                    else:
                        break
                    
                if typeqr != 3:
                    _, bin = cv2.threshold(gray, thres, 255, cv2.THRESH_BINARY)
                    degree = finddirection(bin, number, rotateimgb, box, typeqr)
                    direction.append(degree + 90 - rects[i][2])

    return color, scale, direction, center_all, typeqr

def find_num_box(color, scale, direction, center_all, typeqr):
    
    num_box_all = []

    for i in range(len(direction)):

        center = center_all[i]
        side = scale[i]
        rot_angle = (direction[i]+180) * math.pi/180
        
        if typeqr == 0 or typeqr == 2:

            num_refer_row = np.zeros(2)
            num_refer_row[0] = int(center[0] + (0.65 * side) * math.sin(rot_angle))
            num_refer_row[1] = int(center[1] + (0.65 * side) * math.cos(rot_angle))

            num_0 = np.zeros((1, 2))
            num_0[:, 0] = int(num_refer_row[0] + (0.6 * side) * math.cos(rot_angle))
            num_0[:, 1] = int(num_refer_row[1] - (0.6 * side) * math.sin(rot_angle))

            num_1 = np.zeros((1, 2))
            num_1[:, 0] = int(num_refer_row[0] - (3.9 * side) * math.cos(rot_angle))
            num_1[:, 1] = int(num_refer_row[1] + (3.9 * side) * math.sin(rot_angle))

            num_2 = np.zeros((1, 2))
            num_2[:, 0] = int(num_1[:, 0] + (0.6 * side) * math.sin(rot_angle))
            num_2[:, 1] = int(num_1[:, 1] + (0.6 * side) * math.cos(rot_angle))

            num_3 = np.zeros((1, 2))
            num_3[:, 0] = int(num_0[:, 0] + (0.6 * side) * math.sin(rot_angle))
            num_3[:, 1] = int(num_0[:, 1] + (0.6 * side) * math.cos(rot_angle))
            
            num_box = np.vstack((num_0, num_1, num_2, num_3)).astype(int)

        else:

            num_refer_row = np.zeros(2)
            num_refer_row[0] = int(center[0] + (0.505 * side) * math.sin(rot_angle))
            num_refer_row[1] = int(center[1] + (0.505 * side) * math.cos(rot_angle))

            num_0 = np.zeros((1, 2))
            num_0[:, 0] = int(num_refer_row[0] + (0.645 * side) * math.cos(rot_angle))
            num_0[:, 1] = int(num_refer_row[1] - (0.645 * side) * math.sin(rot_angle))

            num_1 = np.zeros((1, 2))
            num_1[:, 0] = int(num_refer_row[0] - (0.655 * side) * math.cos(rot_angle))
            num_1[:, 1] = int(num_refer_row[1] + (0.655 * side) * math.sin(rot_angle))

            num_2 = np.zeros((1, 2))
            num_2[:, 0] = int(num_1[:, 0] + (0.275 * side) * math.sin(rot_angle))
            num_2[:, 1] = int(num_1[:, 1] + (0.275 * side) * math.cos(rot_angle))

            num_3 = np.zeros((1, 2))
            num_3[:, 0] = int(num_0[:, 0] + (0.275 * side) * math.sin(rot_angle))
            num_3[:, 1] = int(num_0[:, 1] + (0.275 * side) * math.cos(rot_angle))
            
            num_box = np.vstack((num_0, num_1, num_2, num_3)).astype(int)
            num_box_all.append(num_box)

        # img_num = cv2.drawContours(color, [num_box], -1, (0, 255, 255), 5)
        cv2.drawContours(color, [num_box], -1, (0, 255, 255), 5)
    
    return color, num_box_all



def find_antigen_box(color, scale, direction, center_all, typeqr):
    antigen_box_all = []

    for i in range(len(direction)):

        center = center_all[i]
        side = scale[i]
        rot_angle = (direction[i]+180) * math.pi/180

        if typeqr == 0 or typeqr == 2:

            antigen_refer_row = np.zeros(2)
            antigen_refer_row[0] = int(center[0] - (0.875 * side) * math.cos(rot_angle))
            antigen_refer_row[1] = int(center[1] + (0.875 * side) * math.sin(rot_angle))

            antigen_0 = np.zeros((1, 2))
            antigen_0[:, 0] = int(antigen_refer_row[0] - (0.265 * side) * math.sin(rot_angle))
            antigen_0[:, 1] = int(antigen_refer_row[1] - (0.265 * side) * math.cos(rot_angle))

            antigen_1 = np.zeros((1, 2))
            antigen_1[:, 0] = int(antigen_refer_row[0] + (0.265 * side) * math.sin(rot_angle))
            antigen_1[:, 1] = int(antigen_refer_row[1] + (0.265 * side) * math.cos(rot_angle))

            antigen_2 = np.zeros((1, 2))
            antigen_2[:, 0] = int(antigen_1[:, 0] - (2.105 * side) * math.cos(rot_angle))
            antigen_2[:, 1] = int(antigen_1[:, 1] + (2.105 * side) * math.sin(rot_angle))

            antigen_3 = np.zeros((1, 2))
            antigen_3[:, 0] = int(antigen_0[:, 0] - (2.105 * side) * math.cos(rot_angle))
            antigen_3[:, 1] = int(antigen_0[:, 1] + (2.105 * side) * math.sin(rot_angle))
            
            antigen_box = np.vstack((antigen_0, antigen_1, antigen_2, antigen_3)).astype(int)

        else:

            antigen_refer_row = np.zeros(2)
            antigen_refer_row[0] = int(center[0] + (1.175 * side) * math.sin(rot_angle))
            antigen_refer_row[1] = int(center[1] + (1.175 * side) * math.cos(rot_angle))

            antigen_0 = np.zeros((1, 2))
            antigen_0[:, 0] = int(antigen_refer_row[0] + (0.165 * side) * math.cos(rot_angle))
            antigen_0[:, 1] = int(antigen_refer_row[1] - (0.165 * side) * math.sin(rot_angle))

            antigen_1 = np.zeros((1, 2))
            antigen_1[:, 0] = int(antigen_refer_row[0] - (0.165 * side) * math.cos(rot_angle))
            antigen_1[:, 1] = int(antigen_refer_row[1] + (0.165 * side) * math.sin(rot_angle))

            antigen_2 = np.zeros((1, 2))
            antigen_2[:, 0] = int(antigen_1[:, 0] + (1.285 * side) * math.sin(rot_angle))
            antigen_2[:, 1] = int(antigen_1[:, 1] + (1.285 * side) * math.cos(rot_angle))

            antigen_3 = np.zeros((1, 2))
            antigen_3[:, 0] = int(antigen_0[:, 0] + (1.285 * side) * math.sin(rot_angle))
            antigen_3[:, 1] = int(antigen_0[:, 1] + (1.285 * side) * math.cos(rot_angle))
            
            antigen_box = np.vstack((antigen_0, antigen_1, antigen_2, antigen_3)).astype(int)
            antigen_box_all.append(antigen_box)

        # img_num = cv2.drawContours(color, [antigen_box], -1, (255, 255, 0), 5)
        cv2.drawContours(color, [antigen_box], -1, (255, 255, 0), 5)
        
    return color, antigen_box_all

def rotate_with_point(img_in, point, angle):
    cx = int(img_in.shape[1]/2)
    cy = int(img_in.shape[0]/2)
    a1 = np.array([point[1]-cy, point[0]-cx])
    angle_rad = angle*np.pi/180
    rotate_mat = np.array([[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]])
    a2 = np.dot(rotate_mat, a1)
    img_rotate = rotate(img_in, angle)
    h_new, w_new = img_rotate.shape[:2]
    point_rotate = (int(a2[1] + h_new/2), int(a2[0] + w_new/2))
    return img_rotate, point_rotate

def pred_result(img_in):
    img = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    _, img = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY)
    img = img.astype(np.float64) + 1
    seq = 1/np.average(img, axis=0)
    length = seq.shape[0]
    peaks, properties = signal.find_peaks(seq, distance=length//5, prominence=0.005)
    C = 0
    T = 0
    thr = 0.1*length
    for i in peaks:
        if abs(i-length/3) < thr:
            C = 1
        if abs(i-length*2/3) < thr:
            T = 1

    if (C==0 and T==0):
        return 'invalid'
    elif (C==0 and T==1):
        return 'invalid'
    elif (C==1 and T==0):
        return 'negative'
    else:
        return 'positive'

def pred_antigen(img_in, scale, direction, center, typeqr):
    result = None
    if typeqr == 1:
        angle = 270 - direction
        img_rotate, center_rotate = rotate_with_point(img_in, center, angle)
        x1 = int(center_rotate[0] + scale*0.75)
        x2 = int(center_rotate[0] + scale*2.7)
        y1 = int(center_rotate[1] - scale*0.5)
        y2 = int(center_rotate[1] + scale*0.5)
        card_rgb = img_rotate[y1:y2, x1:x2, :]
        width = 512
        height = int(width * card_rgb.shape[0] / card_rgb.shape[1])
        card_rgb = cv2.resize(card_rgb, (width, height))
        card_gray = cv2.cvtColor(card_rgb, cv2.COLOR_BGR2GRAY)
        card_gray = cv2.blur(card_gray, (5, 5))
        card_gray = cv2.Canny(card_gray, 0, 20)
        card_gray = cv2.dilate(card_gray, (15, 15), iterations=20)
        card_gray = cv2.erode(card_gray, (15, 15), iterations=15)
        contours, hierarchy = cv2.findContours(card_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            rect = cv2.minAreaRect(contour)
            center = rect[0]
            w = max(rect[1][0], rect[1][1])
            h = min(rect[1][0], rect[1][1])
            area = w*h
            if (h/w > 0.2 and h/w < 0.3 and area > 20000):
            # if (area > 20000):
                # box = np.int0(cv2.boxPoints(rect))
                # cv2.drawContours(card_rgb, [box], -1, (0, 0, 255), 2)
                x11 = int(center[0]-w/2)
                x22 = int(center[0]+w/2)
                y11 = int(center[1]-h/2)
                y22 = int(center[1]+h/2)
                tmp = card_rgb[y11:y22, x11:x22, :]
                result = pred_result(tmp)
    return result