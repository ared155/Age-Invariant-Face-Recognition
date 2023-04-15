import tkinter as tk
from tkinter.filedialog import askopenfilename
import shutil
import random
import os
import sys
import cv2
##from tqdm import tqdm
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




IMAGE_DIR="AGE_DATA"
window = tk.Tk()

window.title("AGE INVARIANT FACE RECOGNITION SYSTEM")

window.geometry("1500x750")
window.configure(background ="darkcyan")
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
####                        let='D'
##                        r=open('USER_DETAILS.csv', mode = 'r')
##                        rr=r.readlines()
##                        for row in rr:
##                            r1,r2,r3,r4=row.split(",")
##                            if let in r2:
                        r = tk.Label(text="STATUS: MISSING IDENTIFIED", background="darkcyan", fg="Brown", font=("", 15))
                        r.place(x=1000,y=400)
                                    
                        
##                        global a
##                        if a==1:
##                        global let
##                        if let =='B':
####                                        
##                                r = tk.Label(text='STATUS: FACE MATCHED \n CRIMINAL IDENTIFIED..\n AGE : 55\n NAME :BBBBB\n CRIME:SERIAL KILLER', background="darkcyan", fg="Brown", font=("", 15))
##                                r.place(x=1000,y=400)
                        button = tk.Button(text="Exit", command=exit,height=2,width=10,background="#bef062", fg="black", font=("", 15),activebackground="red")
                        button.place(x=1000,y=600)
##                        with open('USER_DETAILS.csv', 'r') as file:
##                                print("Entered")
##                                reader = csv.reader(file)
##                                print(reader)
##                                for row in reader:
##                                        #print(row)
##                                    if let in row:
##                                            print("status is {}".format(row[0]))
##                        if a==0:
##                                        
##                                r = tk.Label(text='STATUS: FACE MATCHED....\n MISSING CHILD IDENTIFIED..', background="darkcyan", fg="Brown", font=("", 15))
##                                r.place(x=1000,y=400)
##                                button = tk.Button(text="Exit", command=exit,height=2,width=10,background="#bef062", fg="black", font=("", 15))
##                                button.place(x=1000,y=600)
                else:
                        print('>face is NOT a Match (%.3f > %.3f)' % (score, thresh))
                        r = tk.Label(text='STATUS: FACE NOT MATCHED....', background="white", fg="black", font=("", 15))
                        r.place(x=1000,y=400)
                        button = tk.Button(text="Exit", command=exit,height=2,width=10,background="#bef062", fg="black", font=("", 15))
                        button.place(x=1000,y=600)
        global fileName1,fileName2,let       
        filenames = [fileName1,fileName2]
        let=fileName1.split('.')[0][-5]
        print(let)
##        if let in 
##        print("the file Name is {}".format(filenames))
        
##        def label_img(img):
##            word_label = img[0]
##            print(word_label)
##        for img in tqdm(os.listdir(IMAGE_DIR)):
##                label = label_img(img)
##                if label==A:
##                        a=1
##                else:
##                        a=0
        # get embeddings file filenames
        embeddings = get_embeddings(filenames)
        is_match(embeddings[0], embeddings[1])
    buttonA = tk.Button(text="ANALYSE", command = main,height=2,width=10,fg="black",bg="#bef062",font=("times",25,"bold"))
    buttonA.place(x=1000,y=200)
buttono = tk.Button(text="OLD PHOTO", command = openphoto,height=2,width=10,fg="black",bg="#bef062",font=("times",25,"bold"))
buttono.place(x=100,y=200)

buttonr = tk.Button(text="RECENT PHOTO", command = openphoto2,height=2,width=15,fg="black",bg="#bef062",font=("times",25,"bold"))
buttonr.place(x=100,y=500)



window.mainloop()




