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
	pix=foto.size/3
	labels = measure.label(blue, neighbors=8, background=0)		
	mask = np.zeros(blue.shape, dtype="uint8")			
	c=0
	for label in np.unique(labels):
		c+=1
		if(c%100==0):
			print (c)					
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
	cv2.imshow("The Mask", cv2.resize(foto, (0,0),fx=hmm, fy=hmm));
	cv2.waitKey(0);
	
	
size=0.05
path="./Pictures/"
path+="konta2.jpg"
hmm=0.4
minval=50
maxval=85
original=cv2.imread(path)
edit=original
edit=cv2.medianBlur(original, 13)
blue=edit
blue=edit[:,:,0]
#tresh=cv2.threshold(blue, sens, 255, cv2.THRESH_BINARY)[1]

#gray=cv2.cvtColor(blue, cv2.COLOR_BGR2GRAY)
gray=blue
tresh=gray
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
