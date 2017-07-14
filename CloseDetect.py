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

def find_hehe(foto, blue):
	mask=blue
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
	#cv2.waitKey(0);
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
		rect = cv2.minAreaRect(cnt)
		carea=rect[1][0]*rect[1][1]
		if(area>int((float(foto.size)/3)*0.01)):
			#cv2.rectangle(foto,(x,y),(x+w,y+h),(255,0,0),15)
			'''
			rect = cv2.minAreaRect(cnt)
			box = cv2.boxPoints(rect)
			box = np.int0(box)
			cv2.drawContours(foto,[box],0,(255,0,0),15)
			print (str(area)+" "+str(carea))
			print (str(int(sfalma(area, carea)*100))+"%")
			cv2.imshow("The Detection by step", cv2.resize(foto, (0,0),fx=hmm, fy=hmm));
			cv2.waitKey(0)
			'''

		if(area>float((float(foto.size)/3)*0.01) and sfalma(area, carea)<0.16):
			neo_cnt.append(cnt)

			rect = cv2.minAreaRect(cnt)
			box = cv2.boxPoints(rect)
			box = np.int0(box)
			cv2.drawContours(foto,[box],0,(0,255,255),15)
			print (str(area)+" "+str(carea))
			print (sfalma(area, carea))

	print (str(count)+":\t"+str(len(neo_cnt)))
	#cv2.drawContours(foto,neo_cnt,-1, (0,0,255), -1)

def inv_colors(img):
	y,x=img.shape
	cop=img.copy()
	for i in range(y):
		for j in range(x):
			cop[i, j]=255-cop[i, j]
	return cop


arxi=0
'''
if(len(sys.argv)>1):
	arxi=int(sys.argv[1])'''
#telos=arxi+1
telos=15
for i in range(arxi, telos):
	count+=1
	path="./Pictures/konta/"
	path+="konta"+str(i+1)+".jpg"
	print (path)
	hmm=0.25
	minval=50
	maxval=85
	original=cv2.imread(path)
	original= cv2.resize(original, (2048, 1536))
	edit=original
	edit=cv2.medianBlur(original, 9)
	edit=adjust_gamma(edit, 0.75)
	edit=cv2.cvtColor(edit, cv2.COLOR_BGR2GRAY)
	cv2.imshow("Aspromavro", cv2.resize(edit, (0,0),fx=hmm, fy=hmm))
	#cv2.waitKey(0)
	edit=cv2.adaptiveThreshold(edit,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,109,-2.1)
	find_hehe(original, edit)
cv2.waitKey(0)
