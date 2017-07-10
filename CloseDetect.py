from skimage import measure
import cv2
import numpy as np
import scipy as sp

def adjust_gamma(image, gamma=1.0):
	invGamma=1.0/gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	return cv2.LUT(image, table) 
def find_hehe(foto, blue):
	size=0.05
	pix=foto.size/3
	labels = measure.label(blue, neighbors=8, background=0)		
	mask = np.zeros(blue.shape, dtype="uint8")			
	c=0
	for label in np.unique(labels):
		c+=1
		if(c%100==0):
			print (label)					
		if label == 0:						
			continue					 
		labelMask = np.zeros(blue.shape, dtype="uint8")		
		labelMask[labels == label] = 255			
		numPixels = cv2.countNonZero(labelMask)
		if (numPixels/pix)*100 > size:				
			mask = cv2.add(mask, labelMask)			
	cv2.imshow("The Mask", cv2.resize(mask, (0,0),fx=hmm, fy=hmm));
	cv2.waitKey(0);
	cnts=cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
	cv2.drawContours(foto,cnts,-1, (0,0,255), -1)
	cv2.imshow("The Detect", cv2.resize(foto, (0,0),fx=hmm, fy=hmm));
	cv2.waitKey(0)
	squ=find_squares(mask)
	for i in squ:
		print (i)
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
			if max_cos < 0.2: 
				squares.append(cnt) 
	return squares 

path="./Pictures/"
path+="konta3.jpg"
hmm=0.4
minval=50
maxval=85
original=cv2.imread(path)
edit=original
edit=cv2.medianBlur(original, 11)
blue=edit
blue=edit[:,:,0]
#tresh=cv2.threshold(blue, sens, 255, cv2.THRESH_BINARY)[1]

#gray=cv2.cvtColor(blue, cv2.COLOR_BGR2GRAY)
grey=blue
tresh=grey
tresh=adjust_gamma(tresh, 0.5)
#tresh=cv2.threshold(gray, sens, 255, cv2.THRESH_BINARY)[1]
'''
for i in range(tresh.shape[0]):
	print (i)
	for j in range(tresh.shape[1]):
		if(minval<=tresh[i, j] and tresh[i, j]<=maxval):
			tresh[i, j]=255
		else:
			tresh[i, j]=0
'''
tresh=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,111,-21)
arr=cv2.resize(tresh, (0,0), fx=hmm, fy=hmm)
#cv2.imshow("Test", arr)
find_hehe(original, tresh)
cv2.waitKey(0)
