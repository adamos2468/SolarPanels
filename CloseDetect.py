import cv2
import numpy as np
import sys

count=0#The count of image proccessed
w=255  #The White Constant
acc=17 #Accepted Error

#A Function to adjust the gamma of a picture
def adjust_gamma(image, gamma=1.0):
	invGamma=1.0/gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	return cv2.LUT(image, table)
#A function that returns the absolute value
def abs(a):
	if(a<0):
		return -a
	return a
#A function that calculates the Error of 2 values
def sfalma(orig, tim):
	return int((abs(float(orig)-float(tim))/orig)*100)

#A function that get the hierarchy of a Picture's contours
#And calculates the number of contours
def NumOfContours(hier):
	#The hierarchy is an 2D array
	#That the 4th value if its -1 then is a White Contour
	c=0
	if not(hier is None):
		for h in hier[0]:
			if(h[3]==-1):
				c+=1
	return c

#A method that detects if a contour intersects
#With any of the other contours
def intersects(cnt, img):
	img=img.copy() 				#Creating the copy of the picture
	gray=deColorStage(img)		#Moving to Grayscale
	#Passing through a threshold
	thresh = cv2.threshold(gray,254,255,cv2.THRESH_BINARY)[1]
	#Getting the hierarchy
	hier = cv2.findContours(thresh,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)[2]
	palia=NumOfContours(hier)	#calculates the number of contours now
	#Draws the contour
	cv2.drawContours(img, [cnt], 0, (w,w,w), -1)
	#Redo the above operation
	gray=deColorStage(img)
	thresh = cv2.threshold(gray,254,255,cv2.THRESH_BINARY)[1]
	hier = cv2.findContours(thresh,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)[2]
	neo=NumOfContours(hier)
	#And if there are less or the same no of contours then it doesnt intersects
	if(neo<=palia):
		return True
	return False

#The main detection Function
def draw_squares(thresh, foto):
	#We detect the contours of the thresholded image
	contours = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[1]
	for cnt in contours: 			#We pass every Contour
		area = cv2.contourArea(cnt)	#I detect the area of the contour
		rect = cv2.minAreaRect(cnt)
		carea=rect[1][0]*rect[1][1] #I calculate the area of the minimum Rect
		if((area>float((float(foto.size)/3)*0.01) and area<=float((float(foto.size)/3)*0.25))):
			#if the area size is accepted (more then 1% and less than 25% of picture)
			if(sfalma(area, carea)<=acc): #And the error is ok
				rect = cv2.minAreaRect(cnt)
				box = cv2.boxPoints(rect)
				box = np.int0(box)
				if (not intersects(cnt, Solars)): #And doesn't intersects
					#Draw it
					cv2.drawContours(foto,[box],0,(b,g,r),10)
					cv2.drawContours(edit, [cnt], 0, (w,w,w), -1)
					cv2.drawContours(Solars, [cnt], 0, (w,w,w), -1)
					#cv2.imshow("Solars", cv2.resize(Solars, (0,0),fx=hmm, fy=hmm))
					#cv2.waitKey(0)
#A method to make a picture black/white
def deColorStage(image):
	image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	return image

#A method to blur the image
def blurStage(image, blur):
	image=cv2.medianBlur(image, blur)
	return image

#A method to change the gamma
def gammaStage(image, gam):
	image=adjust_gamma(image, gam)
	return image

#Doing the addaptive threshhold
def thresholdStage(image, squr, xrw):
	image=cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,squr, xrw)
	return image


arxi=0
telos=15
if(len(sys.argv)>=2 and sys.argv[1]=='-f'):
	arxi=0
	telos=1
#		 [b,     g,   r, blur, squr, xrw,   gam]
modes=[	 [255,   0,   0,    9,  109,  -6,  0.75]
		,[0	 , 255,   0,    3,   69,   2,  0.75]
		,[0  ,   0, 255,    3,   21,   1,  1.75]
		,[0  , 255, 255,    0,   15,   0,   0.3]
		,[255,   0, 255,    3,    5,   0,   0.2]
		]
for i in range(arxi, telos):
	count+=1
	path="./Pictures/konta/"
	path+="konta"+str(i+1)+".jpg" 				#Choosing the image
	if(len(sys.argv)>=2 and sys.argv[1]=='-f'):
		path=sys.argv[2]
	print (path)
	hmm=0.5										#The resize value
	original=cv2.imread(path)					#read the image
	#original= cv2.resize(original, (2048, 1536))#resize it!
	Solars=np.zeros((original.shape), np.uint8)	#the black image of contours
	edit=original.copy()						#the image that marks
	for j in range(len(modes)):					#for each mode
		print ("AT MODE: "+str(j))
		b, g, r, blur, squr, xrw, gam=modes[j] 	#Read the info
		changes=edit.copy()						#Copy the image for changes
		#Pass the image preparation stages
		if(blur>=3):
			changes=blurStage(changes, blur)
		changes=gammaStage(changes, gam)
		changes=deColorStage(changes)
		changes=thresholdStage(changes, squr, xrw)
		#find the solar panels
		draw_squares(changes, original)
	cv2.imshow("Detect: "+str(i+1),  cv2.resize(original, (0,0),fx=hmm, fy=hmm));
	cv2.imwrite("./Ans/CloseDetectResult"+str(i)+".jpg", original);
cv2.waitKey(0)
