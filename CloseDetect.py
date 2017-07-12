from skimage import measure
import cv2
import numpy as np
import scipy as sp
count=0
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
	for label in np.unique(labels):
		if label == 0:
			continue
		labelMask = np.zeros(blue.shape, dtype="uint8")
		labelMask[labels == label] = 255
		numPixels = cv2.countNonZero(labelMask)
		if (numPixels/pix)*100 > size:
			mask = cv2.add(mask, labelMask)
	#Typwmata
	inver=inv_colors(mask)
	cv2.imshow("The Mask", cv2.resize(inver, (0,0),fx=hmm, fy=hmm));
	cv2.waitKey(0);
	draw_squares(inver, foto)
	cv2.imshow("The Detect: "+str(count), cv2.resize(foto, (0,0),fx=hmm, fy=hmm));
	#cv2.waitKey(0)
def abs(a):
	if(a<0):
		return -a
	return a
def sfalma(orig, tim):
	return abs(float(orig)-float(tim))/orig
def draw_squares(thresh, foto):
	bin, contours, hier = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	neo_cnt=[]
	for cnt in contours:
		area = cv2.contourArea(cnt, oriented = False)
		x,y,w,h = cv2.boundingRect(cnt)
		carea=w*h

		if(area>int((float(foto.size)/3)*0.01)):
			cv2.rectangle(foto,(x,y),(x+w,y+h),(255,0,0),15)
			print (str(area)+" "+str(carea))
			print (sfalma(area, carea))
			cv2.imshow("The Detect", cv2.resize(foto, (0,0),fx=hmm, fy=hmm));
			cv2.waitKey(0)

		if(area>float((float(foto.size)/3)*0.01) and sfalma(area, carea)<0.20):
			neo_cnt.append(cnt)
	print (str(count)+":\t"+str(len(neo_cnt)))
	cv2.drawContours(foto,neo_cnt,-1, (0,0,255), -1)

def inv_colors(img):
	y,x=img.shape
	cop=img.copy()
	for i in range(y):
		for j in range(x):
			cop[i, j]=255-cop[i, j]
	return cop
for i in range(10,11):
	count+=1
	path="./Pictures/konta/"
	path+="konta"+str(i+1)+".jpg"
	print (path)
	hmm=0.4
	minval=50
	maxval=85
	original=cv2.imread(path)
	original= cv2.resize(original, (2048, 1536))
	edit=original
	edit=cv2.medianBlur(original, 5)
	blue=edit
	blue=cv2.cvtColor(blue, cv2.COLOR_BGR2GRAY)
	cv2.imshow("Aspromavro", cv2.resize(blue, (0,0),fx=hmm, fy=hmm))
	cv2.waitKey(0)
	#tresh=cv2.threshold(blue, sens, 255, cv2.THRESH_BINARY)[1]
	#gray=cv2.cvtColor(blue, cv2.COLOR_BGR2GRAY)
	grey=blue
	tresh=grey
	tresh=adjust_gamma(tresh, 0.5)
	#tresh=cv2.threshold(gray, sens, 255, cv2.THRESH_BINARY)[1]
	tresh=cv2.adaptiveThreshold(grey,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,109,-12)
	find_hehe(original, tresh)
cv2.waitKey(0)
