import numpy as np
import cv2
import os
import imutils
import math
import time
import asyncio
import mediapipe as mp
###########################
print("rdfgdsd")
#not my code here#
#make another async function for main code part
#instead of while true
#use await sleep to pause execution of main for a while
#then using common object pass the angle
#then continue running!


class pedestrian():
	def __init__(self,location,size,eyes,confidence):
		self.confidence=confidence
		self.location=location #location is a tuple of the form (x,y)
		self.size=size #size of the bounding box
		self.prev_location=[]
		self.gaze=0 #when this is 1, pedestrian is being gazed at!
		self.time=time.time()		
		self.gazetime=0.0
		self.superior=False
		self.eyes=eyes
	def update_eyes(self,location):
		if (not isinstance(location, float)):
			self.eyes=location
	def prioritize(self,bool):
		self.superior=bool
	def addstarttime(self):
		self.starttime=time.time()

	def add_location(self,location):
		if len(self.prev_location)==5:
			for i in range(len(self.prev_location)-1):
				self.prev_location[i]=self.prev_location[i+1]
			self.prev_location[-1]=location
		else:     
			self.prev_location.append(location)
	
	def updategazetime(self):
		if (self.gaze==1):
			#always call updatetime after updategazetime
			self.gazetime=time.time()-self.starttime

	def updatetime(self):
		self.time=time.time()
	
	def changegazestatus(self,status):
		self.gaze=status
	

def issimilar(x,y):
	distance=math.sqrt((x.location[0]-y.location[0])*(x.location[0]-y.location[0])+(x.location[1]-y.location[1])*(x.location[1]-y.location[1]))
	if distance>20:
		return False
	else:
		return True

def priority(x,y):
	#x and y are of pedestrian datatypes
	#if (x>y?)
	if (x.superior==True and y.superior==False):
		return 1
	if (y.superior==True and x.superior==False):
		return 0
	else:
		a1=0
		b1=0
		a2=0
		b2=0
		if (x.size[1]*x.size[0]>y.size[1]*y.size[0]):
			a1=1
			b1=0
			if x.location[1]>=y.location[1]:
				a2=1
				b2=0
			else:
				a2=0
				b2=1
		else:
			a1=0
			b1=1
			if x.location[1]>=y.location[1]:
				a2=1
				b2=0
			else:
				a2=0
				b2=1
		a_net=0.5*(a1+a2)
		b_net=0.5*(b1+b2)
		if (a_net>b_net):
			return 1
		if (a_net<b_net):
			return 0
		if (a_net==b_net):
			return (x.location[1]>y.location[1])


NMS_THRESHOLD=0.1
MIN_CONFIDENCE=0.30	

def pedestrian_detection(image, model, layer_name, personidz=0):
	#image.shape returns as tuple of rows columns color channels
	(H, W) = image.shape[:2]
	results = [] #we store our predictions

	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False) #we create a blob... scale evryy pixel intensity by 1/255, make image of standard size, swap RB channels and dont crop
	model.setInput(blob) #model is yolo trained on yolomini dataset
	layerOutputs = model.forward(layer_name) #output layer; list of list of detections

	boxes = []
	centroids = []
	confidences = []
	

	#above here we store desired output

	for output in layerOutputs: #we loop through the detections
		for detection in output: #for each detection

			scores = detection[5:] #class probabilities (in logits format) begin with index 5, detection= [x,y,h,w,box_score,_,_,_.. diff objects]
			classID = np.argmax(scores) #arg of max scores 
			confidence = scores[classID]	#score of most likely object

			if (classID == personidz) and confidence > MIN_CONFIDENCE: #personidz=0 , correrponds to a person

				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				boxes.append([x, y, int(width), int(height)])
				centroids.append((centerX, centerY))
				confidences.append(float(confidence))
	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idzs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONFIDENCE, NMS_THRESHOLD)
	# ensure at least one detection exists
	if len(idzs) > 0:
		# loop over the indexes we are keeping
		for i in idzs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			# update our results list to consist of the person
			# prediction probability, bounding box coordinates,
			# and the centroid
			res = (confidences[i], (x, y, x + w, y + h), centroids[i])
			results.append(res)
	# return the list of results
	return results

pose_model=mp.solutions.pose
landmarks=mp.solutions.drawing_utils
pose=pose_model.Pose()
labelsPath = "coco.names"
LABELS = open(labelsPath).read().strip().split("\n")
weights_path = "yolov4-tiny.weights"
config_path = "yolov4-tiny.cfg"
model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
layer_name = model.getLayerNames()
layer_name = [layer_name[i-1] for i in model.getUnconnectedOutLayers()]
cap = cv2.VideoCapture("view.mp4")
#cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
writer = None
A=[]
while True:
	gaze_direction=(0,0)
	B=[]
	(grabbed, image) = cap.read()
	if not grabbed:
		break
	image = imutils.resize(image, width=700)
	results = pedestrian_detection(image, model, layer_name, personidz=LABELS.index("person"))
	for i in range(len(results)):
		C=A #c=[p1]
		location=(results[i][2][0],results[i][2][1])
		size=(results[i][1][2]-results[i][1][0],results[i][1][3]-results[i][1][1])
		confidence=results[i][0]
		new_pedestrian=pedestrian(location,size,confidence,(location[0],int(location[1]-size[1]*0.45)))
		focus=new_pedestrian
		focus_top_left=(int(focus.location[0]-size[0]/2),int(focus.location[1]-size[1]/2))
		cropped_pedestrian=image[focus_top_left[1]:focus_top_left[1]+focus.size[1],focus_top_left[0]:focus_top_left[0]+focus.size[0]]
		if(cropped_pedestrian.any()):
			rgbimg= cv2.cvtColor(cropped_pedestrian, cv2.COLOR_BGR2RGB)
			output=pose.process(rgbimg)
			if (output.pose_landmarks!=None):
				lm = output.pose_landmarks
				lmPose  = pose_model.PoseLandmark
				l_inner_eye= (int(lm.landmark[lmPose.LEFT_EYE_INNER].x*cropped_pedestrian.shape[1]),int(lm.landmark[lmPose.LEFT_EYE_INNER].y*cropped_pedestrian.shape[0]))
				r_inner_eye=(int(lm.landmark[lmPose.RIGHT_EYE_INNER].x*cropped_pedestrian.shape[1]),int(lm.landmark[lmPose.RIGHT_EYE_INNER].y*cropped_pedestrian.shape[0]))
				to_gaze=(int((l_inner_eye[0]+r_inner_eye[0])/2),int((l_inner_eye[1]+r_inner_eye[1])/2))
				eye_coordinates=(focus_top_left[0]+to_gaze[0],focus_top_left[1]+to_gaze[1])
				new_pedestrian.update_eyes(eye_coordinates)
		if (len(A)==0):
			A.append(new_pedestrian)
			B.append(0)
	
		for i in range(len(A)):
			if issimilar(new_pedestrian, A[i]):
				if (new_pedestrian.superior):
					A[i].prioritize(True)
				A[i].updategazetime()
				A[i].updatetime()
				A[i].add_location(location)
				A[i].update_eyes(new_pedestrian.eyes)
				B.append(i)
				break
			if (i==len(A)-1):
				#new pedestrian?
				B.append(len(A))
				A.append(new_pedestrian)
		
	C=[]
	for i in range(len(B)):
		C.append(A[B[i]])
	A=C

	#sort according to priority 
	if len(A)>1:
		for i in range(1, len(A)):
			key = A[i]
			j = i-1
			while j >= 0 and priority(key,A[j])==1:
				A[j + 1] = A[j]
				j -= 1
				A[j + 1] = key

	print(len(A))
	#here update the gaze time?
	for i in range(len(A)):
		if (A[i].gaze==1):
			if(A[i].gazetime>2 and i<len(A)-1):
				A[i].changegazestatus(0)
				A[i+1].changegazestatus(1)
				A[i+1].addstarttime()
				focus=A[i+1]
				if (not isinstance(focus.eyes, float)):
					gaze_direction=focus.eyes
					image = cv2.circle(image, focus.eyes, 5, (0,0,255), 5, cv2.FILLED)
				#status.update_angle((yaw_angle,pitch_angle),(status.angle))
				break
			else:
				A[i].updategazetime()
				focus=A[i]
				if (not isinstance(focus.eyes, float)):
					gaze_direction=focus.eyes
					image = cv2.circle(image, focus.eyes, 5, (0,0,255), 5, cv2.FILLED)
				break
		if (i==len(A)-1):
			focus=A[0]
			A[0].changegazestatus(1)
			A[0].addstarttime()
			if (not isinstance(focus.eyes, float)):
					gaze_direction=focus.eyes
					image = cv2.circle(image, focus.eyes, 5, (0,0,255), 5, cv2.FILLED)

	count=1
	for i in range(len(A)):
		text=str(count)
		font = cv2.FONT_HERSHEY_SIMPLEX
		org = (A[i].location[0],A[i].location[1])
		fontScale = 0.5
		thickness=1
		color = (0, 0, 255)
		cv2.putText(image,text, org, font, fontScale, color, thickness, cv2.LINE_AA, False)
		count=count+1

	height=image.shape[0]
	width=image.shape[1]
	x=int(((30*gaze_direction[0])/width)-15)
	y=int(((30*gaze_direction[1])/height)-15)
	if (len(A)==0):
		x=0
		y=0
	cv2.rectangle(image,(int(0.65*image.shape[1]),int(0.75*image.shape[0])), (image.shape[1],image.shape[0]),(255,255,255),-1)
	cv2.circle(image, (int(0.75*image.shape[1]),int(0.85*image.shape[0])) , 30, (0,0,0), 2)
	cv2.circle(image, (int(0.75*image.shape[1]-x),int(0.85*image.shape[0]+y)) , 15, (0,0,0), -1)
	cv2.circle(image, (int(0.9*image.shape[1]),int(0.85*image.shape[0])) , 30, (0,0,0), 2)
	cv2.circle(image, (int(0.9*image.shape[1]-x),int(0.85*image.shape[0]+y)) , 15, (0,0,0), -1)



	#A is the array of pedestrian objects!
	#Do the below task in the above loop!
	#we need to perform a search of pedestrians | Update the location of orignal pedestrians | A to be outside the array 
	# Del the pedestrians now outside the frame | Add new pedestrians | Check gaze Time etc|
		
	mark = cv2.imread('eyecon.png')
	#location,size,confidence
	for res in results:
		cv2.rectangle(image, (res[1][0],res[1][1]), (res[1][2],res[1][3]), (0, 255, 0))
		count=count+1
	cv2.imshow("Detection",image)
	key = cv2.waitKey(1)
	if key == 27:
		break
cap.release()
cv2.destroyAllWindows()
############### now we build a priority system!