# Import required modules
import cv2 as cv
import time
import argparse
def main():
    def getFaceBox(net, frame, conf_threshold=0.7):
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

        net.setInput(blob)
        detections = net.forward()
        bboxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                bboxes.append([x1, y1, x2, y2])
                # cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
        return frameOpencvDnn, bboxes



    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"

    ageProto = "age_deploy.prototxt"
    ageModel = "age_net.caffemodel"

    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    # Load network
    ageNet = cv.dnn.readNet(ageModel, ageProto)
    genderNet = cv.dnn.readNet(genderModel, genderProto)
    faceNet = cv.dnn.readNet(faceModel, faceProto)


    parser = argparse.ArgumentParser(description='Use this script to run age and gender recognition using OpenCV.')
    parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')
    parser.add_argument("--device", default="cpu", help="Device to inference on")

    args = parser.parse_args()


    if args.device == "cpu":
        ageNet.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)

        genderNet.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)
        
        faceNet.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)

        print("Using CPU device")
    elif args.device == "gpu":
        ageNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        ageNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

        genderNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        genderNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

        genderNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        genderNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
        print("Using GPU device")


    # Open a video file or an image file or a camera stream
    cap = cv.VideoCapture(args.input if args.input else 0)
    padding = 20
    while cv.waitKey(1) < 0:
        # Read frame
        t = time.time()
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv.waitKey()
            break

        frameFace, bboxes = getFaceBox(faceNet, frame)
        if not bboxes:
            print("No face Detected, Checking next frame")
            continue

        for bbox in bboxes:
            # print(bbox)
            face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

            blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            # print("Gender Output : {}".format(genderPreds))
            print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))
            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            print("Age Output : {}".format(agePreds))
            # print("AGE: ", age)
            print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))
            # label = "{},{}".format(gender, age)
            # cv.putText(frameFace, label, (bbox[0], bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
            # cv.imshow("Age Gender Demo", frameFace)
            # cv.imwrite("age-gender-out-{}".format(args.input),frameFace)
        print("time : {:.3f}".format(time.time() - t))
    return age, gender


 
# cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=~/opencv_gpu -DINSTALL_PYTHON_EXAMPLES=OFF -DINSTALL_C_EXAMPLES=OFF -DOPENCV_ENABLE_NONFREE=ON -DOPENCV_EXTRA_MODULES_PATH=~/cv2_gpu/opencv_contrib/modules -DPYTHON_EXECUTABLE=~/env/bin/python3 -DBUILD_EXAMPLES=ON -DWITH_CUDA=ON -DWITH_CUDNN=ON -DOPENCV_DNN_CUDA=ON  -DENABLE_FAST_MATH=ON -DCUDA_FAST_MATH=ON  -DWITH_CUBLAS=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.2 -DOpenCL_LIBRARY=/usr/local/cuda-10.2/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda-10.2/include/ ..










# import cv2
# import face_recognition
# import numpy as np
# from keras.models import load_model

# # Load the pre-trained AgeNet model
# age_model = load_model("agenet_model.h5")  # Replace with the path to your pre-trained AgeNet model

# # Load the test image
# test_image = cv2.imread("test_image.jpg")  # Replace with the path to your test image

# # Find faces in the test image
# face_locations = face_recognition.face_locations(test_image)
# num_faces = len(face_locations)

# # Iterate through detected faces
# for i in range(num_faces):
#     top, right, bottom, left = face_locations[i]

#     # Extract the face region from the image
#     face_image = test_image[top:bottom, left:right]

#     # Preprocess the face image for AgeNet
#     face_image = cv2.resize(face_image, (227, 227))
#     face_image = face_image.astype("float") / 255.0
#     face_image = np.expand_dims(face_image, axis=0)

#     # Perform age estimation using AgeNet model
#     age_prediction = age_model.predict(face_image)[0]
#     predicted_age = int(np.argmax(age_prediction))

#     # Draw bounding box and age label on the face
#     cv2.rectangle(test_image, (left, top), (right, bottom), (0, 0, 255), 2)
#     age_label = f"Age: {predicted_age}"
#     cv2.putText(test_image, age_label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

# # Display the resulting image
# cv2.imshow("Facial Recognition with AgeNet", test_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
