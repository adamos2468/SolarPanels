from imutils import contours
from skimage import measure
import numpy as np
import argparse
import imutils
import cv2
import sys
import getopt

gamma=1.6
sens=100
size=0.05

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
	blue=cv2.GaussianBlur(blue, (31,31), 0)				
	blue=adjust_gamma(blue, gamma)					
	blue=cv2.threshold(blue, sens, 255, cv2.THRESH_BINARY_INV)[1]	
	blue=cv2.erode(blue,None,iterations=2)				
	blue=cv2.dilate(blue,None, iterations=4)			
	return blue

def find_failure(foto, crop):
	pix=foto.size/3
	blue=prepare_mask(crop)
	labels = measure.label(blue, neighbors=8, background=0)		
	mask = np.zeros(blue.shape, dtype="uint8")
	for label in np.unique(labels):	
		if(label%100==0):
			print ('label: '+str(label))				
		if label == 0:						
			continue					 
		labelMask = np.zeros(blue.shape, dtype="uint8")		
		labelMask[labels == label] = 255			
		numPixels = cv2.countNonZero(labelMask)
		if (numPixels/pix)*100 > size:				
			mask = cv2.add(mask, labelMask)			
	#cv2.imshow("The Mask", cv2.resize(mask, (0,0),fx=0.4, fy=0.4));
	#cv2.waitKey(0);
	cnts=cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
	cv2.drawContours(foto,cnts,-1, (0,0,255), 10)

####################################__MAIN__###########################################


if(sys.argv[1]=='-b'):
	foto=cv2.imread(sys.argv[2])
	marked=foto.copy()
elif(sys.argv[1]=='-p'):
	foto=cv2.imread(sys.argv[2])
	exept=get_list(sys.argv[3])
	exept=make_good_list(exept)
	marked=make_blue(foto, exept)
find_failure(foto, marked)
cv2.imshow("After Detection", cv2.resize(foto,(0,0),fx=0.4, fy=0.4))
cv2.waitKey(0)
