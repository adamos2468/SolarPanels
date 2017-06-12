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

foto=cv2.imread("Foto.jpg") 					#Diavazw tin fotografia
pix=foto.size/3
cv2.imshow("Normal Picture", cv2.resize(foto,(0,0),fx=0.4, fy=0.4))
cv2.waitKey(0)
typ=foto[:,:,0]							#xechorizw to mple (giati ta fotovoltaika einai mple)
typ=cv2.GaussianBlur(typ, (21,21), 0)				#vazw gaussian
gamma=1.6							#epilegw gamma
typ=adjust_gamma(typ, gamma)					#to kanw apply
sens=102 
typ=cv2.threshold(typ, sens, 255, cv2.THRESH_BINARY_INV)[1]	#to kanw aspro mavro (Osa pixels exoun ligotero apo sens timi gia mple ginonte aspra)
typ=cv2.erode(typ,None,iterations=2)				#liga adjusments gia na figei to noise
typ=cv2.dilate(typ,None, iterations=4)				#liga adjusments gia na figei to noise
labels = measure.label(typ, neighbors=8, background=0)		#Xechorizw tis perioxes me aspro kai exoyn gyro 8 aspra pixels
mask = np.zeros(typ.shape, dtype="uint8")			#Dimiourgw ena mask apo midenika
size=0.05;							#Size of failures

for label in np.unique(labels):					#Pernw apo kathe perioxi
	if label == 0:						#An einai Mideniko tote einai background
		continue					#Kamw Skip 
	labelMask = np.zeros(typ.shape, dtype="uint8")		#Gia kathe perioxi metroume ta pixels
	labelMask[labels == label] = 255			
	numPixels = cv2.countNonZero(labelMask)
	if (numPixels/pix)*100 > size:				#An einai perisotera apo kapio arithmo pixels
		mask = cv2.add(mask, labelMask)			#vazoume ta megala sto mask

cv2.imshow("The Mask", cv2.resize(mask, (0,0),fx=0.4, fy=0.4));
cv2.waitKey(0);

cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,		#xechorizoume tis perioxes san contours
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = contours.sort_contours(cnts)[0]				#sortaroume
 
for (i, c) in enumerate(cnts):					#pernoume apo kathe ena xechorista
	(x, y, w, h) = cv2.boundingRect(c)			#Vriskoume sintetagmenes
	cv2.rectangle(foto,(x,y),(x+w,y+h),(0,0,255),10)	#Zwgrafizoume ta tetragwna

#Typwma ikonas
cv2.imshow("After Detection", cv2.resize(foto,(0,0),fx=0.4, fy=0.4))
cv2.waitKey(0)
