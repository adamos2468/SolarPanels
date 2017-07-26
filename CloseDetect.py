from skimage import measure
import cv2
import numpy as np
import scipy as sp
import sys
count=0
w=255
acc=0.18
def adjust_gamma(image, gamma=1.0):
	invGamma=1.0/gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	return cv2.LUT(image, table)

def abs(a):
	if(a<0):
		return -a
	return a

def sfalma(orig, tim):
	return abs(float(orig)-float(tim))/orig

def find_hehe(foto, mask):
	cv2.imshow("The Mask", cv2.resize(mask, (0,0),fx=hmm, fy=hmm));
	#cv2.waitKey(0)
	draw_squares(mask, foto)

def NumOfContours(hier):
	c=0
	if not(hier is None):
		for h in hier[0]:
			if(h[3]==-1):
				c+=1
	return c

def intersects(cnt, img):
	img=img.copy()
	gray=deColorStage(img)
	thresh = cv2.threshold(gray,254,255,cv2.THRESH_BINARY)[1]
	hier = cv2.findContours(thresh,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)[2]
	palia=NumOfContours(hier)
	cv2.drawContours(img, [cnt], 0, (w,w,w), -1)
	gray=deColorStage(img)
	thresh = cv2.threshold(gray,254,255,cv2.THRESH_BINARY)[1]
	hier = cv2.findContours(thresh,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)[2]
	neo=NumOfContours(hier)
	if(neo<=palia):
		return True
	return False

def draw_squares(thresh, foto):
	contours = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[1]
	neo_cnt=[]
	for cnt in contours:
		area = cv2.contourArea(cnt)
		rect = cv2.minAreaRect(cnt)
		carea=rect[1][0]*rect[1][1]
		if((area>float((float(foto.size)/3)*0.01) and area<=float((float(foto.size)/3)*0.25)) and sfalma(area, carea)<=acc):
			neo_cnt.append(cnt)
			rect = cv2.minAreaRect(cnt)
			box = cv2.boxPoints(rect)
			box = np.int0(box)
			if (not intersects(cnt, Solars)):
				cv2.drawContours(foto,[box],0,(b,g,r),10)
				cv2.drawContours(edit, [cnt], 0, (w,w,w), -1)
				cv2.drawContours(Solars, [cnt], 0, (w,w,w), -1)
				cv2.imshow("Solars", cv2.resize(Solars, (0,0),fx=hmm, fy=hmm))
				#cv2.waitKey(0)

def deColorStage(image):
	image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	return image

def blurStage(image, blur):
	image=cv2.medianBlur(image, blur)
	return image

def gammaStage(image, gam):
	image=adjust_gamma(image, gam)
	return image

def thresholdStage(image, squr, xrw):
	image=cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,squr, xrw)
	return image

arxi=0
telos=15

#		[b,      g,   r, blur, squr, xrw,  gam]
modes=[	 [255,   0,   0,    9,  109,  -6,  0.75]
		,[0	 , 255,   0,    3,   69,   2,  0.75]
		,[0  ,   0, 255,    3,   21,   1,  1.75]
		,[0  , 255, 255,    0,   15,   0,   0.3]
		]
for i in range(arxi, telos):
	count+=1
	path="./Pictures/konta/"
	path+="konta"+str(i+1)+".jpg"
	print (path)
	hmm=0.5
	minval=50
	maxval=85
	original=cv2.imread(path)
	original= cv2.resize(original, (2048, 1536))
	Solars=np.zeros((original.shape), np.uint8)
	edit=original.copy()
	for j in range(len(modes)):
		#cv2.waitKey(0)
		b, g, r, blur, squr, xrw, gam=modes[j]
		changes=edit.copy()
		if(blur>=3):
			changes=blurStage(changes, blur)
		changes=gammaStage(changes, gam)
		changes=deColorStage(changes)
		changes=thresholdStage(changes, squr, xrw)
		find_hehe(original, changes)
	cv2.imshow("Detect: "+str(i+1),  cv2.resize(original, (0,0),fx=hmm, fy=hmm));
cv2.waitKey(0)
