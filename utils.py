import numpy as np
import cv2
import math
from pyrsistent import v
from pyzbar.pyzbar import decode
import scipy.signal as signal
import matplotlib.pyplot as plt
import os


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
    rotated_img = cv2.warpAffine(img, M, (new_w, new_h), borderValue=(255, 255, 255))
    return rotated_img

def qrcode(img):
    code = decode(img)
    # print(code)
    degree = 0
    out = 0

    if len(code) == 0:
        rotate(img,90)
        degree = 90
        code = decode(img)
        #print(degree)
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
                    print("no valid QR found")
    # print(degree)
    typeqr = 0        
    for code_inside in code:
        out = code_inside.data.decode("utf-8")
        print(out)
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
                # print("not a valid type")
    # print(typeqr)
    return out, typeqr, degree

def findtype(number, rotateimg):
    #rotateimg = rotate(rotateimg,rectsi - 90)
    num, typeqrt ,degree = qrcode(rotateimg)
    #print(degree)
    if num == 2021231057:
        typeqrt = 3
    else:
        #direction.append(rectsi - 90 + degree)
        number.append(num)
    return typeqrt

def finddirection(bin, box, typeqr):
    
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

    # print(np.max([s0,s1,s2,s3]))
    
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

    # plt.figure()
    # plt.imshow(img, cmap='gray')
    # plt.show()

    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    squares = []
    areas = []
    rects = []

    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        rect = cv2.minAreaRect(contours[i])
        # print(rect)

        h = max(rect[1][0], rect[1][1])
        w = min(rect[1][0], rect[1][1])
        area_rect = h * w
        if area > 0.6*area_rect and (abs(w - h) < h*0.3) and area_rect < 200000 and area_rect > 10000:
        # if area < 200000 and area > 10000:
            print(area, w/h, area_rect)
            areas.append(area_rect)
            squares.append(contours[i])
            rects.append(rect)

    # cv2.drawContours(color, squares, -1, (255, 0, 0), 5)
    # plt.figure()
    # plt.imshow(color[:, :, [2, 1, 0]])
    # plt.show()

    max_area = max(areas)

    scale = []
    angle = []
    direction = []
    number = []
    center_all = []
    typeqr_all = []

    
    box_qrcode = []
    for i in range(len(squares)):
        if areas[i] > max_area*0.4 and areas[i] < max_area*2:
            print(areas[i])
            box = np.int0(cv2.boxPoints(rects[i]))
            box_qrcode.append(box)
            area_new = cv2.contourArea(box)

            M = cv2.moments(box)
            center = np.zeros(2)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            center[0] = cX
            center[1] = cY
            center = center.astype(int)
            # print(center)
            center_all.append(center)
            cv2.drawContours(color, [box], -1, (255, 0, 0), 5)
        
            scale.append(np.sqrt(area_new))
            angle.append(90-rects[i][2])
            
            rotateimg = np.zeros((2*int(np.around(np.sqrt(area_new))),2*int(np.around(np.sqrt(area_new)))))
            
            cen = np.around(0.5 * box[0] + 0.5 * box[2])
            for ii in range(2*int(np.around(np.sqrt(area_new)))):
                for jj in range(2*int(np.around(np.sqrt(area_new)))):
                    rotateimg[ii][jj] = gray[int(cen[1])-int(np.around(np.sqrt(area_new)))+ii][int(cen[0])-int(np.around(np.sqrt(area_new)))+jj]
            
            _, rotateimgb = cv2.threshold(rotateimg, 60, 255, cv2.THRESH_BINARY)
            # plt.imshow(rotateimgb, 'gray')
            # plt.show()
            # print(rects[i])
            rotateimgb = rotate(rotateimgb, rects[i][2] - 90)
            # 1 is big 0.833, 0 is 0.885
            typeqr = findtype(number, rotateimgb)
            print('TYPE:', typeqr)
            
            thres = 60
            if typeqr == 3:
                print("rnmtq")
            while typeqr == 3:
                print("binary adjust applied")
                if thres < 170:
                    thres = thres + 1
                    print("#####")
                    print(thres)
                    print("#####")
                    typeqr = 1
                    _, rotateimgb = cv2.threshold(rotateimg, thres, 255, cv2.THRESH_BINARY)
                    rotateimgb = rotate(rotateimgb,rects[i][2] - 90)
                    typeqr = findtype(number,rotateimgb)
                else:
                    print("#####")
                    print("binary adjust failed")
                    print("#####")
                    number.append("QR code fail")
                    break
                
            if typeqr != 3:
                _, bin = cv2.threshold(gray, thres, 255, cv2.THRESH_BINARY)
                degree = finddirection(bin, box, typeqr)
                direction.append(degree + 90 - rects[i][2])
                typeqr_all.append(typeqr)
            elif len(direction) != 0:
                direction.append(direction[-1])
                typeqr_all.append(typeqr_all[-1])
            else:
                direction.append(180)
                typeqr_all.append(2)

    print("scale: ", scale)            

    #0-90
    print("angle: ", angle)

    #0-360
    print("direction: ", direction)
    print("number: ", number)

    return color, scale, direction, center_all, typeqr_all, box_qrcode, number

def find_num_box(color, scale, direction, center_all, typeqr_all):
    
    num_box_all = []

    for i in range(len(direction)):

        center = center_all[i]
        typeqr = typeqr_all[i]
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

def find_antigen_box(color, scale, direction, center_all, typeqr_all):
    antigen_box_all = []

    for i in range(len(direction)):

        center = center_all[i]
        typeqr = typeqr_all[i]
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


def rotate_with_point(img_in, points, angle):
    points_rotate = []
    cx = int(img_in.shape[1]/2)
    cy = int(img_in.shape[0]/2)
    img_rotate = rotate(img_in, angle)
    h_new, w_new = img_rotate.shape[:2]
    angle_rad = angle*np.pi/180
    rotate_mat = np.array([[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]])
    for i in range(points):
        point = points[i]
        a1 = np.array([point[1]-cy, point[0]-cx])
        a2 = np.dot(rotate_mat, a1)
        point_rotate = (int(a2[1] + h_new/2), int(a2[0] + w_new/2))
        points_rotate.append(point_rotate)
    return img_rotate, points_rotate


def rotate_point(img, img_r, points, angle):
    cx = int(img.shape[0]/2)
    cy = int(img.shape[1]/2)

    h_new, w_new = img_r.shape[:2]
    angle_rad = angle*np.pi/180
    rotate_mat = np.array([[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]])
    box = np.array(points)
    if box.shape[0] != 2:
        box = box.T
    bias1 = np.array([
        [-cx, -cx, -cx, -cx],
        [-cy, -cy, -cy, -cy]
    ])
    bias2 = np.array([
        [w_new/2, w_new/2, w_new/2, w_new/2],
        [h_new/2, h_new/2, h_new/2, h_new/2]
    ])
    box = box + bias1
    box = box[[1, 0], :]
    box_r = np.matmul(rotate_mat, box)
    box_r = box_r + bias2
    box_r = box_r[[1, 0], :]
    box_minx = int(np.min(box_r.T[:, 0]))
    box_miny = int(np.min(box_r.T[:, 1]))
    box_maxx = int(np.max(box_r.T[:, 0]))
    box_maxy = int(np.max(box_r.T[:, 1]))
    ret = np.array([[box_minx, box_miny], [box_maxx, box_maxy]])
    img_cut = img_r[box_miny:box_maxy, box_minx:box_maxx, :]
    img_valid = 0
    box_valid = 0
    if img_cut.shape[0] > 0 and img_cut.shape[1] > 0:
        img_valid = 1
    if box_minx < box_maxx and box_miny < box_maxy:
        box_valid = 1
    return img_cut, ret, img_valid and box_valid

def split_card(img_in, box_border, box_qrcode, box_paper, box_SN, scale, direction, typeqr):
    if typeqr == 1:
        angle = 270 - direction
    elif typeqr == 2:
        angle = 180 - direction
    else:
        angle = 180 - direction

    img_rotate = rotate(img_in, angle)
    img_paper, box_paper_r, paper_valid = rotate_point(img_in, img_rotate, box_paper, angle)
    img_SN, box_SN_r, SN_valid = rotate_point(img_in, img_rotate, box_SN, angle)
    img_qrcode, box_qrcode_r, qrcode_valid = rotate_point(img_in, img_rotate, box_qrcode, angle)
    img_out, box_border_r, box_border_valid = rotate_point(img_in, img_rotate, box_border, angle)
    if not (paper_valid and SN_valid and box_border_valid and qrcode_valid):
        return None
    box_paper_r[0, :] = box_paper_r[0, :] - box_border_r[0, :]
    box_paper_r[1, :] = box_paper_r[1, :] - box_border_r[0, :]
    box_SN_r[0, :] = box_SN_r[0, :] - box_border_r[0, :]
    box_SN_r[1, :] = box_SN_r[1, :] - box_border_r[0, :]
    box_qrcode_r[0, :] = box_qrcode_r[0, :] - box_border_r[0, :]
    box_qrcode_r[1, :] = box_qrcode_r[1, :] - box_border_r[0, :]
    result = read_result(img_out, box_paper_r, scale)
    SN = OCR_num(img_SN, typeqr)
    card = {
        'img': img_out,
        'box_paper': box_paper_r,
        'box_SN': box_SN_r,
        'result': result,
        'SN': SN,
        'box_qrcode': box_qrcode_r
    }
    return card

def read_result_(img_in):
    img = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    _, img = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY)
    img = img.astype(np.float64) + 1
    seq = 1/np.average(img, axis=0)
    length = seq.shape[0]
    peaks, properties = signal.find_peaks(seq, distance=length//5, prominence=0.005)
    C = 0
    T = 0
    for x in peaks:
        if x > length*0.25 and x < length*0.5:
            C = 1
        if x > length*0.5 and x < length*0.75:
            T = 1
    # thr = 0.1*length
    # for i in peaks:
    #     if abs(i-length/3) < thr:
    #         C = 1
    #     if abs(i-length*2/3) < thr:
    #         T = 1
    # print(peaks)
    # plt.figure()
    # plt.plot(seq)
    # plt.plot(peaks, seq[peaks], 'x')
    # plt.show()  

    if (C==0 and T==0):
        return 'invalid'
    elif (C==0 and T==1):
        return 'invalid'
    elif (C==1 and T==0):
        return 'negative'
    else:
        return 'positive'

def read_result(card_img, box, scale):
    xmin = int(box[0, 0] - scale*0.4)
    xmax = int(box[1, 0] + scale*0.4)
    ymin = int(box[0, 1] - scale*0.4)
    ymax = int(box[1, 1] + scale*0.4)
    tmp_img = card_img[ymin:ymax, xmin:xmax]
    width = 512
    height = int(width * tmp_img.shape[0] / tmp_img.shape[1])
    tmp_img = cv2.resize(tmp_img, (width, height))
    tmp_gray = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2GRAY)
    tmp_gray = cv2.blur(tmp_gray, (5, 5))
    tmp_gray = cv2.Canny(tmp_gray, 0, 30)
    tmp_gray = cv2.dilate(tmp_gray, (15, 15), iterations=20)
    tmp_gray = cv2.erode(tmp_gray, (15, 15), iterations=15)
    contours, hierarchy = cv2.findContours(tmp_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # plt.figure()
    # plt.imshow(tmp_gray, cmap='gray')
    # # plt.imshow(tmp_img[:, :, [2, 1, 0]])
    # plt.show()
    result = 'None'
    for contour in contours:
        try:
            rect = cv2.minAreaRect(contour)
            center = rect[0]
            w = max(rect[1][0], rect[1][1])
            h = min(rect[1][0], rect[1][1])
            area = w*h
            if (h/w > 0.2 and h/w < 0.3 and area > 20000):
                x11 = int(center[0]-w/2)
                x22 = int(center[0]+w/2)
                y11 = int(center[1]-h/2)
                y22 = int(center[1]+h/2)
                tmp = tmp_img[y11:y22, x11:x22, :]
                result = read_result_(tmp)
                # print(result)
                # plt.figure()
                # box = np.int0(cv2.boxPoints(rect))
                # cv2.drawContours(tmp_img, [box], -1, (0, 0, 255), 2)
                # plt.imshow(tmp[:, :, [2, 1, 0]])
                # plt.show()
        except:
            pass
    if result == 'None':
        xmin = int(box[0, 0])
        xmax = int(box[1, 0])
        ymin = int(box[0, 1])
        ymax = int(box[1, 1])
        tmp_img = card_img[ymin:ymax, xmin:xmax]
        result = read_result_(tmp_img)
    return result


def extract_antigen(img_in, antigen_all, direction, typeqr_all):
    img_antigen_all = []

    for i in range(len(antigen_all)):
        num_one = antigen_all[i]
        num_box_x_min = np.min(num_one[:, 0])
        num_box_x_max = np.max(num_one[:, 0])
        num_box_y_min = np.min(num_one[:, 1])
        num_box_y_max = np.max(num_one[:, 1])

        # print(num_box_x_min, num_box_x_max, num_box_y_min, num_box_y_max)
        
        

        img_antigen = img_in[num_box_y_min:num_box_y_max, num_box_x_min:num_box_x_max]

        typeqr = typeqr_all[i]
        if typeqr == 1:
            rot_angle = -direction[i]-90
        else:
            rot_angle = -direction[i]+180
        img_antigen_rotate = rotate(img_antigen, rot_angle)
        img_antigen_all.append(img_antigen_rotate)

    return img_antigen_all

def find_antigen(color, scale, direction, center_all, typeqr_all):
    antigen_all = []

    for i in range(len(direction)):

        center = center_all[i]
        typeqr = typeqr_all[i]
        side = scale[i]
        rot_angle = (direction[i]+180) * math.pi/180

        if typeqr == 0 or typeqr == 2:

            antigen_refer_row = np.zeros(2)
            antigen_refer_row[0] = int(center[0] + (1.425 * side) * math.sin(rot_angle))
            antigen_refer_row[1] = int(center[1] + (1.425 * side) * math.cos(rot_angle))

            antigen_0 = np.zeros((1, 2))
            antigen_0[:, 0] = int(antigen_refer_row[0] + (3.225 * side) * math.cos(rot_angle))
            antigen_0[:, 1] = int(antigen_refer_row[1] - (3.225 * side) * math.sin(rot_angle))

            antigen_1 = np.zeros((1, 2))
            antigen_1[:, 0] = int(antigen_refer_row[0] - (5.525 * side) * math.cos(rot_angle))
            antigen_1[:, 1] = int(antigen_refer_row[1] + (5.525 * side) * math.sin(rot_angle))

            antigen_2 = np.zeros((1, 2))
            antigen_2[:, 0] = int(antigen_1[:, 0] - (2.755 * side) * math.sin(rot_angle))
            antigen_2[:, 1] = int(antigen_1[:, 1] - (2.755 * side) * math.cos(rot_angle))

            antigen_3 = np.zeros((1, 2))
            antigen_3[:, 0] = int(antigen_0[:, 0] - (2.755 * side) * math.sin(rot_angle))
            antigen_3[:, 1] = int(antigen_0[:, 1] - (2.755 * side) * math.cos(rot_angle))
            
            antigen_box = np.vstack((antigen_0, antigen_1, antigen_2, antigen_3)).astype(int)

        else:

            antigen_refer_row = np.zeros(2)
            antigen_refer_row[0] = int(center[0] - (0.855 * side) * math.sin(rot_angle))
            antigen_refer_row[1] = int(center[1] - (0.855 * side) * math.cos(rot_angle))

            antigen_0 = np.zeros((1, 2))
            antigen_0[:, 0] = int(antigen_refer_row[0] + (0.875 * side) * math.cos(rot_angle))
            antigen_0[:, 1] = int(antigen_refer_row[1] - (0.875 * side) * math.sin(rot_angle))

            antigen_1 = np.zeros((1, 2))
            antigen_1[:, 0] = int(antigen_refer_row[0] - (0.875 * side) * math.cos(rot_angle))
            antigen_1[:, 1] = int(antigen_refer_row[1] + (0.875 * side) * math.sin(rot_angle))

            antigen_2 = np.zeros((1, 2))
            antigen_2[:, 0] = int(antigen_1[:, 0] + (5.805 * side) * math.sin(rot_angle))
            antigen_2[:, 1] = int(antigen_1[:, 1] + (5.805 * side) * math.cos(rot_angle))

            antigen_3 = np.zeros((1, 2))
            antigen_3[:, 0] = int(antigen_0[:, 0] + (5.805 * side) * math.sin(rot_angle))
            antigen_3[:, 1] = int(antigen_0[:, 1] + (5.805 * side) * math.cos(rot_angle))
            
            antigen_box = np.vstack((antigen_0, antigen_1, antigen_2, antigen_3)).astype(int)
            
        antigen_all.append(antigen_box)

        # img_num = cv2.drawContours(color, [antigen_box], -1, (255, 255, 0), 5)
        cv2.drawContours(color, [antigen_box], -1, (0, 255, 0), 5)
        
    return color, antigen_all

def row_segment(img):
    row_seg = []
    h, w = img.shape
    row_his = np.zeros((h, 1))
    for i in range(h):
        for j in range(w):
            if img[i, j] == 0:
                row_his[i, :] = row_his[i, :] + 1
    for i in range(h-1):
        if row_his[i] == 0 and row_his[i+1] != 0:
            row_seg.append(i)
        if row_his[i] != 0 and row_his[i+1] == 0:
            row_seg.append(i)
    
    if len(row_seg) == 0:
        row_seg = [0, h]
    if len(row_seg) == 1:
        if row_seg[0] < h/2:
            row_seg.append(h)
        if row_seg[0] > h/2:
            row_seg.append(0)
            row_seg[1] = row_seg[0]
            row_seg[0] = 0
    if len(row_seg) > 2:
        row_mask = np.zeros(int(len(row_seg)/2))
        for i in range(int(len(row_seg)/2)):
            row_mask[i] = row_seg[2*i+1]-row_seg[2*i]
        mask_max = np.argmax(row_mask)
        row_seg_new = []
        row_seg_new.append(row_seg[2*mask_max])
        row_seg_new.append(row_seg[2*mask_max+1])
        row_seg = row_seg_new
    # print(row_seg)
    
    # plt.plot(row_his)
    # plt.show()

    return row_seg, row_his

def extract_num(img_in, num_box_all, direction):
    img_num_all = []

    for i in range(len(num_box_all)):
        num_one = num_box_all[i]
        num_box_x_min = np.min(num_one[:, 0])
        num_box_x_max = np.max(num_one[:, 0])
        num_box_y_min = np.min(num_one[:, 1])
        num_box_y_max = np.max(num_one[:, 1])

        # print(num_box_x_min, num_box_x_max, num_box_y_min, num_box_y_max)
        
        img_num = img_in[num_box_y_min:num_box_y_max, num_box_x_min:num_box_x_max]

        rot_angle = -direction[i]+180
        img_num_rotate = rotate(img_num, rot_angle)
        img_num_all.append(img_num_rotate)
        
        # cv2.imwrite('./img_report/yuan'+str(i)+'.png', img_num)
        # cv2.imwrite('./img_report/zhuan'+str(i)+'.png', img_num_rotate)

        # plt.imshow(img_num)
        # plt.show()

        plt.imshow(img_num_rotate)
        plt.show()

    return img_num_all

def OCR_num(img, typeqr):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_baw = cv2.threshold(img_gray, 110, 255, cv2.THRESH_BINARY)
    # plt.imshow(img_baw, 'gray')
    # plt.show()
    ocr_fail = 0
    row_seg, row_his = row_segment(img_baw)
    # print(np.sum(row_his))
    if np.sum(row_his) < 3000:
        ocr_fail = 1
    # print(row_his.shape)
    # plt.imshow(img)
    # plt.show()
    if ocr_fail == 0:
        img_row_seg = img_gray[row_seg[0]:row_seg[1], :]
        # plt.imshow(img_row_seg, 'gray')
        h_row, w = img_row_seg.shape
        col_seg = []
        baw_hold = 150
        
        if typeqr == 1:
            while len(col_seg) != 12*2:
                col_seg = []
                baw_hold = baw_hold - 5
                # print(baw_hold)
                _, img_baw = cv2.threshold(img_row_seg, baw_hold, 255, cv2.THRESH_BINARY)
                col_his = np.zeros((w, 1))
                for i in range(w):
                    for j in range(h_row):
                        if img_baw[j, i] == 0:
                            col_his[i, :] = col_his[i, :] + 1
                for i in range(w-1):
                    if col_his[i] == 0 and col_his[i+1] != 0:
                        col_seg.append(i)
                    if col_his[i] != 0 and col_his[i+1] == 0:
                        col_seg.append(i)
                for i in range(int(len(col_seg)/2)):
                    if col_seg[2*i+1]-col_seg[2*i] < 5:
                        del col_seg[2*i]
                        del col_seg[2*i]
                        break
                    if col_seg[2*i+1]-col_seg[2*i] > 45:
                        col_seg.insert(2*i+1, int(np.round((col_seg[2*i+1]+col_seg[2*i])/2)))
                        col_seg.insert(2*i+1, int(np.round((col_seg[2*i+2]+col_seg[2*i])/2)))
                if baw_hold == 10:
                    ocr_fail = 1
                    # print('OCR failure')
                    break
                # print(len(col_seg))
        else:
            while len(col_seg) != 15*2:
                col_seg = []
                baw_hold = baw_hold - 5
                # print(baw_hold)
                _, img_baw = cv2.threshold(img_row_seg, baw_hold, 255, cv2.THRESH_BINARY)
                col_his = np.zeros((w, 1))
                for i in range(w):
                    for j in range(h_row):
                        if img_baw[j, i] == 0:
                            col_his[i, :] = col_his[i, :] + 1
                for i in range(w-1):
                    if col_his[i] == 0 and col_his[i+1] != 0:
                        col_seg.append(i)
                    if col_his[i] != 0 and col_his[i+1] == 0:
                        col_seg.append(i)
                for i in range(int(len(col_seg)/2)-1):
                    if col_seg[2*i+1]-col_seg[2*i] < 5:
                        # print(2*i, 2*i+1)
                        del col_seg[2*i]
                        del col_seg[2*i]
                        break
                    if col_seg[2*i+1]-col_seg[2*i] > 100:
                        col_seg.insert(2*i+1, int(np.round((col_seg[2*i+1]+col_seg[2*i])/2)))
                        col_seg.insert(2*i+1, int(np.round((col_seg[2*i+2]+col_seg[2*i])/2)))
                    if col_seg[2*i+2]-col_seg[2*i+1] < 60:
                        if i <= 4:
                            del col_seg[2*i+1]
                            del col_seg[2*i+1]
                            break
                if baw_hold == 10:
                    ocr_fail = 1
                    # print('OCR failure')
                    break
                # print(len(col_seg))
    if ocr_fail == 1:
        ocr_result = 'OCR failure'
    else:
        num_all = []
        # print(int(len(col_seg)/2))
        for i in range(int(len(col_seg)/2)):
            num_all.append(img_baw[:, col_seg[2*i]:col_seg[2*i+1]])
        # for i in range(len(num_all)):
        #     plt.imshow(num_all[i], 'gray')
        #     plt.show()
        for i in range(len(num_all)):
            img_num_one = num_all[i]
            row_num_one, _ = row_segment(img_num_one)
            num_all[i] = img_num_one[row_num_one[0]:row_num_one[1], :]
            num_all[i] = cv2.resize(num_all[i], (25, 30))
            _, num_all[i] = cv2.threshold(num_all[i], 110, 255, cv2.THRESH_BINARY)
            
            # cv2.imwrite('./img_num/'+str(i)+'.tif', num_all[i])
            # plt.imshow(num_all[i], 'gray')
            # plt.show()
        img_temp_all = []
        temp_name = []
        file_pathname = 'template'
        for filename in os.listdir(file_pathname):
        #     print(filename)
            temp_name.append(filename)
            img_temp = cv2.imread('./'+file_pathname+'/'+filename)
            img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)
            _, img_temp = cv2.threshold(img_temp, 110, 255, cv2.THRESH_BINARY)
            img_temp_all.append(img_temp)
        list = []
        img_dif = np.zeros(len(img_temp_all))
        for i in range(len(num_all)):
            for j in range(len(img_temp_all)):
                img_dif[j] = np.sum(abs(num_all[i] - img_temp_all[j]))
            mask = np.argmin(img_dif)
            list.append(temp_name[mask].split('_')[0])
        if typeqr == 1:
            for i in range(len(list)):
                if list[i] == 'O' or list[i] == 'C':
                    list[i] = '0'
        else:
            for i in range(len(list)):
                if list[i] == 'O' or list[i] == 'C':
                    if i > 6:
                        list[i] = '0'
            list[0] = '东'
            list[1] = '方'
            list[2] = '基'
            list[3] = '因'
        ocr_result = ''.join(list)
    return ocr_result