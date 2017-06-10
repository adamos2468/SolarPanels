from imutils import contours
from skimage import measure
import numpy as np
import argparse
import imutils
import cv2

def adjust_gamma(image, gamma=1.0):
	invGamma=1.0/gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	return cv2.LUT(image, table)	

fx=0.4
fy=0.4
foto=cv2.imread("Foto.jpg") 	#Diavazw tin fotografia
cv2.imshow("Normal Picture", cv2.resize(foto,(0,0),fx=0.4, fy=0.4))
cv2.waitKey(0)


typ=foto[:,:,0]						#xechorizw to kokkino
typ=cv2.GaussianBlur(typ, (21,21), 0)			#vazw gaussian
gamma=1.6						#epilegw gamma
typ=adjust_gamma(typ, gamma)				#to kanw apply
typ=cv2.threshold(typ, 100, 255, cv2.THRESH_BINARY_INV)[1]	#to kanw aspro mavro
typ=cv2.erode(typ,None,iterations=2)			#liga adjusments
typ=cv2.dilate(typ,None, iterations=4)			#liga adjusments

labels = measure.label(typ, neighbors=8, background=0)
mask = np.zeros(typ.shape, dtype="uint8")

for label in np.unique(labels):
	# if this is the background label, ignore it
	if label == 0:
		continue
 
	# otherwise, construct the label mask and count the
	# number of pixels 
	labelMask = np.zeros(typ.shape, dtype="uint8")
	labelMask[labels == label] = 255
	numPixels = cv2.countNonZero(labelMask)
 
	# if the number of pixels in the component is sufficiently
	# large, then add it to our mask of "large blobs"
	if numPixels > 1500:
		mask = cv2.add(mask, labelMask)

cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = contours.sort_contours(cnts)[0]
 
# loop over the contours
for (i, c) in enumerate(cnts):
	# draw the bright spot on the image
	(x, y, w, h) = cv2.boundingRect(c)
	((cX, cY), radius) = cv2.minEnclosingCircle(c)
	#cv2.circle(foto, (int(cX), int(cY)), int(radius),
	#	(0, 0, 255), 10)
	cv2.rectangle(foto,(x,y),(x+w,y+h),(0,0,255),10)
 
# show the output image
cv2.imshow("After Detection", cv2.resize(foto,(0,0),fx=0.4, fy=0.4))
cv2.waitKey(0)
#############################################
##################Typwma#####################
#############################################
#typ=cv2.resize(typ,(0,0),fx=0.25, fy=0.25)
#cv2.imshow("Fotovoltaika Testing with Gamma: "+str(i), typ)
#cv2.waitKey(0)
