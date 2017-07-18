from skimage import measure
import cv2
import numpy as np
import scipy as sp
import sys

count=0
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

def auto_canny(image, sigma=0.55):
    v = np.median(image)
    print(sigma,"   ",v,"   ")
    temp = (1.0 - sigma) * v
    if temp<0:
        lower = 0
    else:
        lower = int(temp)
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged

def find_hehe(foto, mask):
	#cv2.imshow("The Mask", cv2.resize(mask, (0,0),fx=hmm, fy=hmm));
	draw_squares(mask, foto)

def draw_squares(thresh, foto):
	bin, contours, hier = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	neo_cnt=[]
	for cnt in contours:
		area = cv2.contourArea(cnt)
		rect = cv2.minAreaRect(cnt)
		carea=rect[1][0]*rect[1][1]
		if(area>float((float(foto.size)/3)*0.01) and sfalma(area, carea)<=0.1):
			neo_cnt.append(cnt)
			rect = cv2.minAreaRect(cnt)
			box = cv2.boxPoints(rect)
			box = np.int0(box)
			cv2.drawContours(foto,[box],0,(b,g,r),15)
	cv2.drawContours(edit,neo_cnt,-1, (255,255,255), -1)

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

#		[b,   g,   r, blur, squr, xrw,  gam]
modes=[	[255, 0	 , 0,    9,  109,  -7, 0.75],
		[0	, 255, 0,    3,   69,   2,  0.5]
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
	edit=original.copy()
	for j in range(len(modes)):
		#cv2.imshow("The Edit", cv2.resize(edit, (0,0),fx=hmm, fy=hmm));
		#cv2.waitKey(0)
		b, g, r, blur, squr, xrw, gam=modes[j]
		changes=edit.copy()
		if(blur>=3):
			changes=blurStage(changes, blur)
		changes=gammaStage(changes, gam)
		changes=deColorStage(changes)
		changes=thresholdStage(changes, squr, xrw)
		find_hehe(original, changes)
	cv2.imshow("Detect: "+str(count),  cv2.resize(original, (0,0),fx=hmm, fy=hmm));
cv2.waitKey(0)
