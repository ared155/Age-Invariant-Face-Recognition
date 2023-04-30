import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter.filedialog import askopenfilename
import shutil
import random
import os
import sys
import cv2
import os
from PIL import Image, ImageTk
import cv2
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import sys
import csv
global fileName1,fileName2

class App:
    def __init__(self, window, window_title, video_source):
        self.window = window
        self.window.title(window_title)
        
        # Open the video source
        self.cap = cv2.VideoCapture(video_source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Create a canvas that can fit the video source
        self.canvas = tk.Canvas(window, width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), 
                                height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()
        
        # Use PIL (Pillow) to convert the OpenCV image to a Tkinter image
        self.photo = None
        self.update()
        
        dirPath = "testpicture"
        fileList = os.listdir(dirPath)
        for fileName in fileList:
                os.remove(dirPath + "/" + fileName)


        def openphoto():
            global fileName1
            
            # C:/Users/sagpa/Downloads/images is the location of the image which you want to test..... you can change it according to the image location you have  
            fileName1 = askopenfilename(initialdir='C:\\Users\\LENOVO\\Desktop\\2023 PROJECTS\\AGE_INVARIANT', title='Select image for analysis ',
                                   filetypes=[('image files', '.jpg')])
            
            dst = "testpicture"
            print(fileName1)
            print (os.path.split(fileName1)[-1])
            if os.path.split(fileName1)[-1].split('.') == 'h (1)':
                print('dfdffffffffffffff')
            shutil.copy(fileName1, dst)
            load1 = Image.open(fileName1)
            im1=load1.resize((300,300), Image.ANTIALIAS)
            render = ImageTk.PhotoImage(im1)
            img = tk.Label(image=render, height="300", width="300")
            img.image = render
            img.place(x=500, y=75)
        ##    buttono.configure(image=img)
        ##    img.grid(column=0, row=1, padx=10, pady = 10)
        def openphoto2():
            global fileName2
            # C:/Users/sagpa/Downloads/images is the location of the image which you want to test..... you can change it according to the image location you have  
            fileName2 = askopenfilename(initialdir='C:\\Users\\LENOVO\\Desktop\\2023 PROJECTS\\AGE_INVARIANT', title='Select image for analysis ',
                                   filetypes=[('image files', '.jpg')])
            dst = "testpicture"
            print(fileName2)
            print (os.path.split(fileName2)[-1])
            if os.path.split(fileName2)[-1].split('.') == 'h (1)':
                print('dfdffffffffffffff')
            shutil.copy(fileName2, dst)
            load2 = Image.open(fileName2)
            im2=load2.resize((300,300), Image.ANTIALIAS)
            render = ImageTk.PhotoImage(im2)
            img1 = tk.Label(image=render, height="300", width="300")
            img1.image = render
            img1.place(x=500, y=425)

        ##    buttonr.configure(image=img1)
            def main():
        ##    img.grid(column=0, row=1, padx=10, pady = 10)
                def extract_face(filename, required_size=(224, 224)):
                        # load image from file
                        pixels = pyplot.imread(filename)
                        # create the detector, using default weights
                        detector = MTCNN()
                        # detect faces in the image
                        results = detector.detect_faces(pixels)
                        # extract the bounding box from the first face
                        x1, y1, width, height = results[0]['box']
                        x2, y2 = x1 + width, y1 + height
                        print("FACE DETECTED.....")
                        # extract the face
                        face = pixels[y1:y2, x1:x2]
                        # resize pixels to the model size
                        image = Image.fromarray(face)
                        image = image.resize(required_size)
                        face_array = asarray(image)
                        return face_array
                
                def get_age_gender(filename):
                    # extract face
                    face = extract_face(filename)
                    # convert face to an array of samples
                    samples = asarray([face], 'float32')
                    # prepare the face for the model, e.g. center pixels
                    samples = preprocess_input(samples, version=2)
                    # create a vggface model for age prediction
                    age_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
                    # predict age
                    age = age_model.predict(samples)
                    # create a vggface model for gender prediction
                    gender_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
                    # predict gender
                    gender = gender_model.predict(samples)
                    # calculate cosine distance between age and gender vectors
                    dist = cosine(age, gender)
                    # return age and gender
                    return age, gender, dist

                 
                # extract faces and calculate face embeddings for a list of photo files
                def get_embeddings(filenames):
                        # extract faces
                        faces = [extract_face(f) for f in filenames]
                        # convert into an array of samples
                        samples = asarray(faces, 'float32')
                        # prepare the face for the model, e.g. center pixels
                        samples = preprocess_input(samples, version=2)
                        # create a vggface model
                        model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
                        # perform prediction
                        yhat = model.predict(samples)
                        return yhat
                 
                # determine if a candidate face is a match for a known face
                def is_match(known_embedding, candidate_embedding, thresh=0.5):
                        # calculate distance between embeddings
                        score = cosine(known_embedding, candidate_embedding)
                        if score <= thresh:
                                print('>face is a Match (%.3f <= %.3f)' % (score, thresh))
                                r = tk.Label(text="STATUS: MISSING IDENTIFIED", background="darkcyan", fg="Brown", font=("", 15),bg="#3b1d7d")
                                r.place(x=1000,y=400)

                                button = tk.Button(text="Exit", command=exit,height=2,width=10,background="#3b1d7d", fg="black", font=("", 15),activebackground="red")
                                button.place(x=1000,y=600)
                        else:
                                print('>face is NOT a Match (%.3f > %.3f)' % (score, thresh))
                                r = tk.Label(text='STATUS: FACE NOT MATCHED....', background="white", fg="black", font=("", 15))
                                r.place(x=1000,y=400)
                                button = tk.Button(text="Exit", command=exit,height=2,width=10,background="#3b1d7d", fg="black", font=("", 15))
                                button.place(x=1000,y=600)
                global fileName1,fileName2,let       
                filenames = [fileName1,fileName2]
                let=fileName1.split('.')[0][-5]
                print(let)
                # get embeddings file filenames
                embeddings = get_embeddings(filenames)
                age, gender, dist = get_age_gender(filenames[1])
                if (dist<0.5):
                      print('Gender:Male')
                else :
                      print('Gender: Female')
                # print('Age:',age)
                is_match(embeddings[0], embeddings[1])
            buttonA = tk.Button(text="ANALYSE", command = main,height=1,width=10,fg="black",bg="#3b1d7d",font=("times",15,"bold"))
            buttonA.place(x=1000,y=200)
        buttono = tk.Button(text="OLD PHOTO", command = openphoto,height=1,width=10,fg="black",bg="#3b1d7d",font=("times",15,"bold"))
        buttono.place(x=100,y=200)

        buttonr = tk.Button(text="RECENT PHOTO", command = openphoto2,height=1,width=15,fg="black",bg="#3b1d7d",font=("times",15,"bold"))
        buttonr.place(x=100,y=500)



        window.mainloop()


        # Start the video playback loop
        self.window.mainloop()
    
    def update(self):
        # Get a frame from the video source
        ret, frame = self.cap.read()
        if ret:
            self.photo = ImageTk.PhotoImage(image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)
        
        # Repeat after 15 milliseconds
        self.window.after(15, self.update)

# Create a window and pass it to the Application object
App(tk.Tk(), "Tkinter Video Looping Background", "video.mp4")

