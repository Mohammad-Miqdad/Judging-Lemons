import numpy as np
from PIL import Image
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
import cv2
import tensorflow as tf
import serial
import time

# Open the serial port
ser = serial.Serial('/dev/cu.usbmodem14101', 9600)
time.sleep(2)  # Wait for the serial connection to initialize

# Load the base MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Load the trained classifier model
classifier = tf.keras.models.load_model('citrus_classifier.h5')

# Define a function to preprocess the captured images
def preprocess_image(filename):
    with Image.open(filename) as image:
        image = image.resize((224, 224))  # Resize the image to the size used in training
        image = np.array(image)  # Convert the image to a numpy array
        image = preprocess_input(image)  # Preprocess the image using MobileNetV2's preprocessing function
    return image


# Connect to the cameras
cams = [cv2.VideoCapture(i) for i in range(2)]

# Continuously capture and classify images
while True:
    frames = []
    for i, cam in enumerate(cams):
        ret, frame = cam.read()
        cv2.imshow(f'frame{i}', frame)
        frames.append(frame)

    # Check for key press
    key = cv2.waitKey(1000)
    if key == ord('c'): # Capture and classify the images
        predictions = []
        for i, frame in enumerate(frames):
            # Capture the images
            cv2.imwrite(f'capture{i}.jpg', frame)

            # Preprocess the captured images
            image = preprocess_image(f'capture{i}.jpg')

            # Extract the features from the image using the base model
            features = base_model.predict(image[np.newaxis, ...])

            # Flatten the features
            flat_features = features.reshape(1, -1)

            # Classify the images
            prediction = classifier.predict(flat_features)

            # Get the class with the highest probability
            class_prediction = np.argmax(prediction)

            # Store the prediction in the list
            predictions.append(class_prediction)

            # Print the predictions
            print(f'Prediction for camera {i}:', class_prediction)

            # After the loop, check if both predictions are class 1
            if all(pred == 0 for pred in predictions):
                print("Both images are healthy.")
                # Send a command to the Arduino
                ser.write(b'2') # Command to rotate arm by ninety degrees
                time.sleep(2)  # Delay for 2 seconds

                # Take another set of images and predict
                for i, frame in enumerate(frames):

                    # Same code used here to classify image as figure 13


                    if all(pred == 0 for pred in predictions):
                        print("Both iamges are healthy.")
                        # Send a command to the Arduino
                        ser.write(b'1') # Command to move lemon to healthy pile
                        time.sleep(2)  # Delay for 2 seconds

                    else:
                        print("At least one of the images is defective.")
                        ser.write(b'0') # Command to move lemon to defective pile
                        time.sleep(2)  # Delay for 2 seconds

            else:
                print("At least one of the images is defective.")
                ser.write(b'0') # Command to move lemon to defective pile
                time.sleep(2)  # Delay for 2 seconds


    elif key == ord('q'):  # Quit the program
        break

# Release the cameras and close the windows
for cam in cams:
    cam.release()
cv2.destroyAllWindows()

# Close the serial port
ser.close()