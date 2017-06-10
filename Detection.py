from __future__ import print_function
import numpy as np
import argparse
import cv2
import sys
import glob
from PIL import Image
import math
import argparse
import imutils

# im = cv2.imread("/home/john/Desktop/photo/DJI_0186.JPG") # estiatoria
im = cv2.imread("/home/john/Desktop/photo/DJI_0035.JPG") # fos apo psila
#im = cv2.imread("/home/john/Desktop/photo/fot.JPG") # komati me ilio
#im = cv2.imread("/home/john/Desktop/photo/DJI_0168.JPG")
# im = cv2.imread("/home/john/Desktop/photo/DJI_0014.JPG") # parko mikro

h_f, width_f, n = im.shape


# rec function global vairable
original = im
img = im
visited = []
flag_frame = 0
max = 0
ypos = 0
sys.setrecursionlimit(1000000000)



def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def prepro(original):
    all = []
    for gamma in np.arange(0.0, 1.6, 0.5):
        # ignore when gamma is 1 (there will be no change to the image)
        if gamma == 1:
            continue

        # apply gamma correction and show the images
        gamma = gamma if gamma > 0 else 0.1
        adjusted = adjust_gamma(original, gamma=gamma)
        cv2.putText(adjusted, "g={}".format(gamma), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

        gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)

        size = 2

        cv2.imwrite("/home/john/Desktop/photos/gammaa"+str(gamma)+".jpg", adjusted)


        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        # blurred = gray
        # apply Canny edge detection using a wide threshold, tight
        # threshold, and automatically determined threshold
        wide = cv2.Canny(blurred, 10, 200)
        wide = cv2.dilate(wide, np.ones((size, size)))
        squares = find_squares(wide)
        cv2.imwrite("/home/john/Desktop/photos/gamma1.jpg", wide)



        tight = cv2.Canny(blurred, 225, 250)
        tight = cv2.dilate(tight, np.ones((size, size)))
        squares2 = find_squares(tight)
        cv2.imwrite("/home/john/Desktop/photos/gamma2.jpg", tight)


        auto = auto_canny(blurred)
        auto = cv2.dilate(auto, np.ones((size, size)))
        squares3 = find_squares(auto)
        cv2.imwrite("/home/john/Desktop/photos/gamma3.jpg", auto)



        for i in range(0, len(squares)):
            all.append(squares.pop(0))
        for i in range(0, len(squares2)):
            all.append(squares2.pop(0))
        for i in range(0, len(squares3)):
            all.append(squares3.pop(0))


    return all

def find_white(x,y,lengthy):
    for i in range(int(y-(lengthy/2)),int(y+(lengthy/2))):
        if im[i][x] == 255:
            return 1
        return 0

def maxim(x, y):
    if x > y:
        return x
    return y

def minim(x, y):
    if x < y:
        return x
    return y

def in_list(x,y):
    global visited
    length = len(visited)
    for i in range(0,length):
        if(visited[i][0] == x and visited[i][1] == y):
            return 1
    return 0

def sort(monadika):
    dimension = 0

    for i in range(len(monadika)):
        dimension = (monadika[i][1] - monadika[i][0]) + dimension

    dimension = int(dimension / len(monadika))
    i = 0


    sorted = []
    miny_pos = 0
    flag_min_pos = 0
    while len(monadika) != 0:  # sort all frames
        miny = monadika[0][2]
        miny_pos = 0
        flag_min_pos = 0


        for j in range(0, len(monadika)):  # find the minY
            if miny > monadika[j][2]:
                miny = monadika[j][2]
                miny_pos = j
                flag_min_pos = 1

        temp = monadika[0]

        miny = miny + int(dimension / 2)
        s = []
        sorted_temp = []
        j = 0
        flag = 1
        flag_exit = 0

        while (flag == 1):  # sort monadika in temp by y
            if miny < monadika[j][3] and miny > monadika[j][2]:
                temp = monadika.pop(j)

                sorted_temp.append(temp)
                flag_exit = 1
            else:
                j = j + 1
            if (j == len(monadika)):
                flag = 0

        # sorted_temp has all y frames
        temp2 = []

        while (len(sorted_temp) != 0):  # sort temp in temp2 by x
            i_pos = 0
            temp = sorted_temp[0]
            for i in range(0, len(sorted_temp)):  # find minx
                if temp[0] > sorted_temp[i][0]:
                    temp = sorted_temp[i]
                    i_pos = i

                    flag_exit = 1
            temp2 = sorted_temp.pop(i_pos)

            sorted.append(temp2)
            # extra kodikas gia diplotipa
        if flag_exit == 0 and miny_pos < len(monadika):  # an den yparxei plaisio stn a3ona y
            monadika.pop(miny_pos)
    return sorted

def rec(pos, y, length):
    global max
    global visited
    global flag_frame
    global ypos
    global h_f

    if (pos < 0 or y==h_f or y==0):
        return
    if length == pos:
        flag_frame = 1
        ypos = y
        return

    if in_list(pos, y) == 1 and len(visited) != 0:
        return
    else:
        temp = []
        temp.append(pos)
        temp.append(y)
        visited.append(temp)


    if (flag_frame == 0 and im[y][pos - 1]) == 255:
        rec(pos - 1, y, length)
    if (flag_frame == 0 and im[y + 1][pos]) == 255:
        rec(pos, y + 1, length)
    if flag_frame == 0 and im[y + 1][pos - 1] == 255:
        rec(pos - 1, y + 1, length)
    if flag_frame == 0 and im[y - 1][pos - 1] == 255:
        rec(pos - 1, y - 1, length)

def rec2(pos, y,length):

    global max
    global visited
    global flag_frame
    global ypos
    global h_f
    global width_f

    if (pos == width_f-1 or y==0 or y==h_f-1):
        max = width_f-1
        return

    if length == pos :
        flag_frame = 1
        ypos = y

    if(pos == length):
        return

    if in_list(pos,y) == 1 and len(visited) != 0:
        return
    else:
        temp = []
        temp.append(pos)
        temp.append(y)
        visited.append(temp)

    max = maxim(pos, max)


    if (flag_frame == 0 and im[y][pos + 1]) == 255:
        rec2(pos + 1, y,length)
    if(flag_frame == 0 and im[y+1][pos]) == 255:
        rec2(pos,y+1,length)
    if flag_frame == 0 and im[y+1][pos + 1] == 255:
        rec2(pos+1, y+1,length)
    if flag_frame == 0 and im[y-1][pos + 1] == 255:
        rec2(pos +1, y-1,length)

def find_lines(pos,y,length,flag):
    global visited
    global flag_frame
    global max
    global ypos

    visited = []
    flag_frame = 0
    max = 0
    ypos = 0

    if flag == 1:
        rec2(pos,y,length)
    elif flag == 0:
        rec(pos,y,length)

def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def print_list(list):
    print(len(list[0]),"\n")
    print(list[0])
    for i in range(0,len(list)):
        print(list[i])
    print("\n")

def exist(t,all):
    for i in range(0,len(all)):
        if same_range(t[0],t[1],all[i][0],all[i][1]) == 1 and same_range(t[2],t[3],all[i][2],all[i][3]) == 1:
            return 1
    return 0

def same_range( x1,x2,nx,nx2 ):
    if (nx2-nx) > ((x2-x1)*1.5): # orthogonia apo dipla plaisia epistrefei false
        return 0

    if (x1 >= nx and x1 <=nx2) or (x2 >= nx and x2 <=nx2) or (nx >= x1 and nx <=x2) or (nx2 >= x1 and nx2 <=x2):
        return 1

    return 0

def same_in_x (x1,x2,pos):
    if(pos >= x1 and pos <= x2):
        return 1
    return 0

def find_squares(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    squares = []
    for gray in cv2.split(img):
        for thrs in range(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            bin,contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)
    return squares

def write_and_crop(list):
    if len(list) == 0 :
        print("Empty List")
        return 0

    temp = list[0]
    string = "/home/john/Desktop/plaisia/Frame"

    dimensionx = int(dimension / 20)  # 5%
    dimensiony = int(dimension / 22)  # 4.5%
    for i in range(0, len(list)):
        str_path = (string + str(i+1) + ".jpg")
        cropped = im[list[i][2] - dimensionx:list[i][3] + dimensionx,
                  list[i][0] - dimensiony:list[i][1] + dimensiony]
        cv2.imwrite(str_path, cropped)

    print("Number of frames :",i+1)

def extra(list):
    #print_list(list)
    if len(list) == 0 :
        return
    temp = []
    y = []

    dimension = int((list[0][3] - list[0][2])/2)

    y = dimension + list[0][2]

    count = 0
    for i in range(0,len(list)):
        if y < list[i][3] and y > list[i][2] :
            count = count +1
            temp.append(list[i])
    for j in range(0,count):
        list.pop(0)

    i_temp = []
    i=0
    av = 0
    if len(temp) != 0:
        for i in range(0,len(temp)) :
            av = av + (temp[i][1] - temp[i][0])
        av = int(av/(i+1))
    else:
        return []


    # print("len(list) : ",len(list), "   ")
    # print("av : ",av)
    # print("extra temp : ",temp)
    # print("extra len(temp) : ",len(temp))

    wrong_pos = []
    correct = []

    for i in range(0,len(temp)):
        if av*1.1 > (temp[i][1]-temp[i][0]) and av/2 < (temp[i][1]-temp[i][0]) and temp[i][1]-temp[i][0] > int(av*0.5):
            correct.append(temp[i])


    temp = correct

    # print("len(wrong) : ",len(wrong_pos))



    return temp

        # list2 = []
        # op_flag = 1
        #
        #
        # while (flag_frame == 1 and op_flag == 1):
        #     find_lines(temp2[0], temp2[3], temp2[0] - avg1, 0)
        #     if (flag_frame == 1):
        #         x2 = list[0][0] - int((list[0][1] - list[0][0]) * 0.1)
        #         x1 = list[0][0] - int(avg * 1.1)
        #         y1 = list[0][2]
        #         y2 = ypos
        #         temp2 = []
        #         temp2.append(x1)
        #         temp2.append(x2)
        #         temp2.append(y1)
        #         temp2.append(y2)
        #         temp2.append(1)
        #         temp2.append(x1 - int(avg / 2))
        #
        #         if (exist(temp2, list) or exist(temp2, final)):
        #             op_flag = 0
        #         else:
        #             list2.insert(0, temp2)
        #
        #         temp2 = list2[0]
        # for l in range(0, len(list2)):
        #     list.append(list2[l])

def mark_frames(list):
    global original
    print("number of frames : ",len(list))
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(0, len(list)):
        temp = list[i]
        x = (int((temp[1]-temp[0])/2) + temp[0])
        y = (int((temp[3]-temp[2])/2) + temp[2])

        im = cv2.circle(original, (x, y), 10, (255, 255, 255), -1)  # kitrino

        cv2.circle(im, (temp[0], temp[2]), 5, cv2.QT_FONT_BLACK, -1)
        cv2.circle(im, (temp[0], temp[3]), 5, cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, -1)
        cv2.circle(im, (temp[1], temp[2]), 5, (255, 0, 0), -1)
        cv2.circle(im, (temp[1], temp[3]), 5, 255, -1)

def find_lost_frames (list):
    temp = []
    temp = list[0]
    length = list [0][1] - list [0][0]
    lengthy = list[0][3] - list[0][2]

    flag = 0
    temp2 = []
    count = 0

    font = cv2.FONT_HERSHEY_SIMPLEX
    av = 0
    k=0

    for k in range(0,len(list)):
        av = av + (list[k][1] - list[k][0])
    av = int(av/k)

    if(list[0][0] - int(av/2) > av): # lost frames in x -
        pos = list[0][0] - int(av / 2)
        print("----------------- ",av)
        #pos = pos - 12
        while(pos > 0):

            for i in range(0,len(list)):
                temp = list[i]
                if( same_in_x(temp[0],temp[1],pos)):
                    flag = 1

            if(flag == 0):
               x2 = pos + int(length/2)
               x1 = pos - int(length/2)
               print("x1 is : ",x1)
               print("x2 is : ",x2)
               if(x1>=0) :
                temp2.append(x1)
                temp2.append(x2)
                temp2.append(list[0][2])
                temp2.append(list[0][3])
                count = count +1

                cv2.putText(im, "#",
                            (x1 + int(length/2) ,int((list[0][3]-list[0][2])/2)+list[0][2]), font, 1,
                            (0, 0, 0), 8)

            pos = pos - av
            flag = 0
            print("pos is : ", pos)


    else : # lost frames in x +
        pos = list[0][1] + int(av/2)

        print("--------av--------- ", av)

        h, width, n = im.shape

        cur_pos = 0
        k = 2
        while (pos < (width-av/2)):

            print("\n\n",pos,"\n\n" ,k)


            for i in range(0, len(list)): # forward frames and find lost

                temp = list[i]
                cur_pos = (temp[1]) + int(av / 2)

                if (same_in_x(temp[0], temp[1], pos) and flag==0):
               #     print("yparxei : ", k)
                    print("cur   :   ",cur_pos,"  ",temp)
                    pos = cur_pos
                    flag = 1
                    #pos = pos + (temp[1] - temp[0])


            if (flag == 0):
               # print("k is : ", k)
                x2 = pos + int(length / 2) # new position
                x1 = pos - int(length / 2)
                print("x1 is : ", x1)
                print("x2 is : ", x2)

                if (x1 < width):#################################
                    temp2.append(x1)
                    temp2.append(x2)
                    temp2.append(list[0][2])
                    temp2.append(list[0][3])
                    count = count + 1

                    cv2.putText(im, "#",
                                (x1 + int(length / 2), int((list[0][3] - list[0][2]) / 2) + list[0][2]), font, 1,
                                (0, 0, 0), 8)
                    pos = pos + x2-x1

            flag = 0
            print("pos is : ", pos )
            k = k+1


    print("count is  : ",count)

def lost(list,final):
    global max
    global flag_frame
    global visited
    global ypos
    plus = 0
    list_temp = []

    font = cv2.FONT_HERSHEY_SIMPLEX
    avg = 0

    i = 0
    for i in range(0,len(list)):
        avg = avg + (list[i][1] - list[i][0])


    avg = int(avg/(i+1))
    avg1 = avg - int(avg * 0.4)

    lost= []
    temp = list[0]

    flag_pixel = 1

    while(flag_pixel == 1):


        find_lines(list[0][0], list[0][3], list[0][0] - avg1, 0) # aristera

        if(flag_frame == 1):
            x2 = list[0][0] - int((list[0][1]-list[0][0]) * 0.1)
            x1 = list[0][0] - int(avg*1.1)
            y1 = list[0][2]
            y2 = ypos
            temp_pos = []
            temp_pos.append(x1)
            temp_pos.append(x2)
            temp_pos.append(y1)
            temp_pos.append(y2)
            temp_pos.append(1)
            temp_pos.append(x1 - int(avg/2))

            if x1 > 0 and exist(temp_pos,list) == 0 and exist(temp_pos,final) == 0:
                list.insert(0,temp_pos)
            else :
                flag_pixel=0
        else :
            flag_pixel = 0

    i=0
    # length = len(list)
    while i < len(list):

        temp2 = []
        flag_frame = 1
        temp2 = list[i]
        temp_pos = []
        if(temp[1] + avg1) < list[i][0]: # an den yparxei plaisio meta
            print("is in and i is :",i)
            find_lines(temp[1],temp[3],temp[1]+avg1,1)
            print("flag ",flag_frame)

            if(flag_frame == 1): #an yparxei gramh
                x1 = temp[1] + int(avg * 0.1)
                x2 = temp[1] + int(avg*1.1)
                y1 = ypos - (temp[3] - temp[2])
                y2 = ypos

                temp_pos.append(x1)
                temp_pos.append(x2)
                temp_pos.append(y1)
                temp_pos.append(y2)
                temp2.append(1)
                temp2.append(x2 - int(avg / 2))

                if (exist(temp_pos,list) == 0 ) and (exist(temp2,final) == 0):
                    list.insert(i,temp_pos)




        if (list[i][0] - avg1) > temp[1]:

            find_lines(list[i][0], list[i][3], list[i][0] - avg1, 0)

            if (flag_frame == 1):  # an yparxei gramh

                x2 = list[i][0] - int(avg * 0.10)#temp[1] + int(avg * 0.10)
                x1 = list[i][0] - int(list[i][1] - list[i][0])#temp[1] + avg
                y1 = ypos - (temp[3] - temp[2])
                y2 = ypos

                temp_pos = []
                temp_pos.append(x1)
                temp_pos.append(x2)
                temp_pos.append(y1)
                temp_pos.append(y2)
                temp_pos.append(1)
                temp_pos.append(x2 + int(avg / 2))

                if exist(temp_pos, list) == 0 and exist(temp_pos,final) == 0:
                    list.insert(i, temp_pos)




        temp = list[i]
        i = i + 1


    flag_frame = 1

    while(flag_frame == 1):
        size = len(list)
        find_lines(list[size-1][1], list[size-1][3], list[size-1][1] + avg1, 1)
        print("this is the new test---------->    ",flag_frame)
        if (flag_frame == 1):
            x2 = list[size-1][1] + int(avg * 1.1)
            x1 = list[size-1][1] + int(avg * 0.1)
            y1 = ypos - (list[size-1][3] - list[size-1][2])
            y2 = ypos
            temp_pos = []
            temp_pos.append(x1)
            temp_pos.append(x2)
            temp_pos.append(y1)
            temp_pos.append(y2)


            if x2 < width_f and exist(temp_pos, list) == 0 and exist(temp_pos, final) == 0:
                list.append(temp_pos)
            else:
                flag_frame = 0

    return

def duplicate(list_all):
    monadika = []
    flag = 1
    dimension = 0

    for i in range(len(list_all)):
        dimension = (list_all[i][1] - list_all[i][0]) + dimension

    dimension = int(dimension / len(list_all))
    i = 0

    while flag:  # megala plaisia
        avx = (list_all[i][1] - list_all[i][0])
        if (avx > int(dimension * 1.5)) or (avx < int(dimension * 0.4)):
            list_all.pop(i)
        else:
            i = i + 1

        if i == len(list_all):
            flag = 0

    flag = 1

    for i in range(0, len(list_all)):
        flag = 1
        temp = list_all.pop()

        for j in range(0, len(monadika)):  # if already exist
            if same_range(temp[0], temp[1], monadika[j][0], monadika[j][1]) and same_range(temp[2], temp[3],
                                                                                           monadika[j][2],
                                                                                           monadika[j][3]):
                if monadika[j][0] > temp[0]:
                    monadika[j][0] = temp[0]
                if monadika[j][2] > temp[2]:
                    monadika[j][2] = temp[2]
                if monadika[j][1] < temp[1]:
                    monadika[j][1] = temp[1]
                if monadika[j][3] > temp[3]:
                    monadika[j][3] = temp[3]

                flag = 0
        if flag == 1:
            monadika.append(temp)

    return monadika

def auto_canny(image, sigma=0.55):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    print(sigma,"   ",v,"   ")
    # apply automatic Canny edge detection using the computed median
    temp = (1.0 - sigma) * v
    if temp<0:
        lower = 0
    else:
        lower = int(temp)
    # lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

def positions(squares):
    list_all = []
    for j in range(0, len(squares)):  # get min and max of x,y of all frames
        temp = squares.pop(0)
        xmax = 0
        xmin = temp[0][0]
        ymax = 0
        ymin = temp[0][1]
        i = 0

        for numx in temp:

            if xmax < numx[0]:
                xmax = numx[0]
            if xmin > numx[0]:
                xmin = numx[0]
            if ymax < numx[1]:
                ymax = numx[1]
            if ymin > numx[1]:
                ymin = numx[1]

        list = []
        list.append(xmin)
        list.append(xmax)
        list.append(ymin)
        list.append(ymax)
        list.append(0)
        list.append(0)

        list_all.append(list)
    return list_all

def angle (a, b, c):
    return math.degrees(math.acos((c**2 - b**2 - a**2)/(-2.0 * a * b)))

def rotation(original):
    img = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)  # gray
    img = cv2.blur(img, (3, 3))

    kernel = np.ones((5, 5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    img = cv2.Canny(img, 50, 200)

    img = cv2.dilate(img, np.ones((5, 5)))
    t, img = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY)


    pos_list = find_squares(img)
    pos_list = positions(pos_list)
    print(len(pos_list),"  ->before")



    print(len(pos_list),"  ->after")

    pos_list = duplicate(pos_list)
    pos_list = sort(pos_list)

    max_pos = []
    next = []
    max_len = 0;

    while(True):
        next = extra(pos_list)
        if(len(next)>max_len):
            max_len=len(next)
            max_pos=next

        if(len(pos_list)==0):
            break
    print(len(max_pos),"\t",max_pos)
    one=[]
    two=[]
    three=[]

    print(max_pos[0],max_pos[len(max_pos)-1])
    if(len(max_pos)>1):
        one.append(max_pos[0][0])
        one.append(max_pos[0][3])
        two.append(max_pos[len(pos_list)-1][1])
        two.append(max_pos[len(pos_list)-1][3])

        three.append(two[0])
        three.append(one[1])

    ypotinousa= math.sqrt(pow((two[0]-one[0]),2)+pow((two[1]-one[1]),2))
    platos = math.sqrt(pow((three[0]-one[0]),2)+pow((three[1]-one[1]),2))
    ipsos = math.sqrt(pow((two[0]-three[0]),2)+pow((two[1]-three[1]),2))

    num = int(angle(ypotinousa, platos, ipsos)+1)

    print(num," This is a num")
    rotated = []
    if(one[1]>two[1]):
        rotated = imutils.rotate_bound(original,num)
    else:
        rotated = imutils.rotate_bound(original,-num)

        # h,width,n = rotated.shape
        # print(rotated.shape)




    cv2.namedWindow('blacke', cv2.WINDOW_NORMAL)
    cv2.imshow("blacke", im)
    cv2.waitKey(0)


    cv2.namedWindow('rotated', cv2.WINDOW_NORMAL)
    cv2.imshow("rotated", rotated)
    cv2.waitKey(0)


    return rotated



################################################# main #################################################################
if __name__ == '__main__':

        cv2.imwrite("/home/john/Desktop/photos/original.jpg", original)
        # original = imutils.rotate_bound(original, 3)

        cv2.namedWindow('original', cv2.WINDOW_NORMAL)
        cv2.imshow("original", original)
        cv2.waitKey(0)

        original = rotation(original)

        original2 = original

        img = original2
        im = original2



        # img = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) # gray
        img = cv2.blur(img, (3, 3))

        kernel = np.ones((5, 5), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

        img = cv2.Canny(img, 50 ,200)





        img = cv2.dilate(img, np.ones((5, 5)))
        t, img = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY)

        im = img

        cv2.namedWindow('black', cv2.WINDOW_NORMAL)
        cv2.imshow("black", im)
        cv2.waitKey(0)





        squares = prepro(original)

        cv2.drawContours(original2, squares, -1, (0, 255, 0), 1) # DRAW



        cv2.namedWindow('black', cv2.WINDOW_NORMAL)
        cv2.imshow("black", im)
        cv2.waitKey(0)
        # cv2.imwrite("/home/john/Desktop/photos/black.jpg", im)

        cv2.namedWindow('normal', cv2.WINDOW_NORMAL)
        cv2.imshow("normal", original2)
        cv2.waitKey(0)
        # cv2.imwrite("/home/john/Desktop/photos/found_frames.jpg", original2)




        #komeni = im[1036:1274,18:165]
        #komeni = im[1020:1285, 185:335]

        #img = cv2.circle(img,(165, 1030), 20, (0,255,255),-1) # kitrino
        #img = cv2.circle(img, (169, 1270), 20, (255, 0, 255), -1)
        #img = cv2.circle(img, (24, 1274), 20, (255, 255, 0), -1)
        #img = cv2.circle(img, (18, 1036), 20, (255, 0, 255), -1) #lila

        #img = cv2.circle(img,(185, 1026), 20, (0,255,255),-1) # kitrino
        # img = cv2.circle(img, (169, 1270), 20, (255, 0, 255), -1)
        # img = cv2.circle(img, (24, 1274), 20, (255, 255, 0), -1)
        # img = cv2.circle(img, (18, 1036), 20, (255, 0, 255), -1) #lila




        new = img
        cv2.namedWindow('pro', cv2.WINDOW_NORMAL)
        cv2.imshow("pro", img)


        list_all = []
        list=[]

        list_all = positions(squares) #change the types of positions

        monadika = []
        print("Nubrer of frames found : ",len(list_all))
        monadika = duplicate(list_all)
        # flag = 1
        # dimension = 0
        #
        # for i in range(len(list_all)):
        #     dimension = (list_all[i][1] - list_all[i][0]) + dimension
        #
        # dimension = int(dimension/len(list_all))
        # i=0
        #
        # while flag: # megala plaisia
        #     avx = (list_all[i][1] - list_all[i][0])
        #     if (avx > int(dimension*1.5)) or (avx < int(dimension* 0.4)) :
        #         list_all.pop(i)
        #     else :
        #         i = i+1
        #
        #     if i==len(list_all) :
        #         flag = 0
        #
        #
        # flag = 1
        #
        # for i in range(0, len(list_all)):
        #     flag = 1
        #     temp = list_all.pop()
        #
        #     for j in range(0, len(monadika)): # if already exist
        #         if same_range(temp[0], temp[1], monadika[j][0], monadika[j][1]) and same_range(temp[2], temp[3], monadika[j][2], monadika[j][3]):
        #             if monadika[j][0] > temp[0]:
        #                  monadika[j][0] = temp[0]
        #             if monadika[j][2] > temp[2]:
        #                  monadika[j][2] = temp[2]
        #             if monadika[j][1] < temp[1]:
        #                  monadika[j][1] = temp[1]
        #             if monadika[j][3] > temp[3]:
        #                  monadika[j][3] = temp[3]
        #
        #             flag = 0
        #     if flag == 1 :
        #         monadika.append(temp)


        sorted = sort(monadika)

        i=0
        c = 4

        print(len(sorted) , "sorted")


        final = []

        while(len(sorted)!=0):
            print("this is the value of integer i : ",i)
            print("\n\n")
            print("sorted  ", len(sorted) , "   i : ",i+1)
            t = []
            t = extra(sorted)                #lost(t)
            print("t : ",len(t))


            print("sorted  ",len(sorted) , "   i : ",i+1)
            if (i+1) == c:
                print()
            print_list(t)
            lost(t, final)
            mark_frames(t)

            print("len list     ",len(list))


            for l in range(0,len(t)):
                final.append(t[l])


            i = i+1


        cv2.namedWindow('marked', cv2.WINDOW_NORMAL)
        cv2.imshow("marked", original)
        cv2.waitKey(0)
        cv2.imwrite("/home/john/Desktop/photos/final_with_lost_frames.jpg",original)