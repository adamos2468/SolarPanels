from imutils import contours
from skimage import measure
import numpy as np
import argparse
import imutils
import cv2
import sys
import getopt

gamma=1.6
sens=116
blur=35
hmm=1
def adjust_gamma(image, gamma=1.0):
	invGamma=1.0/gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	return cv2.LUT(image, table)

def make_good_list(now):
	neo=[]
	for i in now:
		neo.append(i.split().copy())
		for j in range(0,4):
			neo[-1][j]=int(neo[-1][j])
	return neo

def make_blue(image, exept):
	cop=image.copy()
	cop[:,:,0]=255
	for i in exept:
		for g in range(i[2], i[3]):
			for q in range(i[0], i[1]):
				cop[q][g]=foto[q][g].copy()
	return cop

def get_list(path):
	fil=open(path, 'r')
	lis=fil.readlines()
	fil.close()
	return lis

def prepare_mask(foto):
	blue=foto[:,:,0]
	blue=cv2.GaussianBlur(blue, (blur, blur), 0)
	blue=adjust_gamma(blue, gamma)
	blue=cv2.threshold(blue, sens, 255, cv2.THRESH_BINARY_INV)[1]
	blue=cv2.erode(blue,None,iterations=2)
	blue=cv2.dilate(blue,None, iterations=4)
	return blue

def find_failure(foto, crop):
	pix=foto.size/3
	mask=prepare_mask(crop)
	cnts=cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
	neo_arr=[]
	for cnt in cnts:
		print (cnt)
		area=cv2.contourArea(cnt)
		x,y,w,h = cv2.boundingRect(cnt)
		if(area>=pix*0.01):
			neo_arr.append(cnt)
	cv2.drawContours(foto,neo_arr,-1, (0,0,255), 3)

####################################__MAIN__###########################################
path="./Pictures/"
path+="broken1.jpg"
ogiginal=cv2.imread(path)
if(sys.argv[1]=='-b'):
	original=cv2.imread(sys.argv[2])
original=cv2.resize(original, (500, 900))
marked=original.copy()
find_failure(original, marked)
cv2.imshow("After Detection", cv2.resize(original,(0,0),fx=hmm, fy=hmm))
cv2.waitKey(0)
