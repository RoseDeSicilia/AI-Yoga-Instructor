# In progress: code does not work
# code to read images, take their landmarks and export to csv as training data
import os, os.path # import os python library to deal with file path directory and operations
import numpy as np
import cv2

def create():
# create folder Training_Poses to hold all types of correct yoga poses to get their landmarks as training data
	if not "Training_Poses" in os.listdir("."):
		os.mkdir("Training_Poses")
	else:
		return
	
	yogaPose = cv2.CascadeClassifier('.xml') # research cascade classifiers for poses

	label = 0
	i=1
	arr = []
# Training_Data folder: will hold the images for Testing Data / Validation data split
	for dirname, dirnames, filenames in os.walk('Training_Data'):

		for subdirname in dirnames:

			subject_path = os.path.join(dirname, subdirname)

			for filename in os.listdir(subject_path):
# loop through all types of image formats / files in the folder(s)
				if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):

					abs_path = "%s/%s" % (subject_path, filename)
					image=cv2.imread(abs_path)
					poses = yogaPose.detectMultiScale(image,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30),flags = 0)

					for (x, y, w, h) in poses:
						os.chdir("Training_Poses")
						cv2.imwrite(str(label)+str(i)+".jpg",image[y-15:y+h+15,x-15:x+w+15])
						arr.append( ["Training_Poses/"+str(label)+str(i)+".jpg",label] )
						os.chdir("../")
						cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
						i+=1
					np.savetxt('coords.csv',arr,delimiter=',', fmt='%s')

			label = label + 1

	print ("csv data appended success!")
			
