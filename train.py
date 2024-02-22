import os
import numpy as np
from PIL import Image
from keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
import random
from keras.utils import to_categorical
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.models import Sequential


random.seed(42)

# Set up the data
data_dir = r'C:\Users\Mohammad Yasoob\PycharmProjects\pythonProject\dataset\lemons'
image_filenames = []
labels = []

# Create a LabelEncoder object
label_encoder = LabelEncoder()

# Loop through each subfolder (healthy or defective) in the fruit folder
for subfolder_name in os.listdir(data_dir):
    subfolder_dir = os.path.join(data_dir, subfolder_name)

    # Check if it is a directory
    if os.path.isdir(subfolder_dir):

        # If it is a defective subfolder, loop through each image file
        if subfolder_name == 'defective':
            for filename in os.listdir(subfolder_dir):
                if filename.endswith('.jpg'):
                    labels.append('defective')
                    image_filenames.append(os.path.join(subfolder_dir, filename))

        # If it is a healthy subfolder, loop through each image file
        elif subfolder_name == 'healthy':
            for filename in os.listdir(subfolder_dir):
                if filename.endswith('.jpg'):
                    labels.append('healthy')
                    image_filenames.append(os.path.join(subfolder_dir, filename))


# Shuffle the dataset
shuffled_indices = np.random.permutation(len(image_filenames))
image_filenames = np.array(image_filenames)[shuffled_indices]
labels = np.array(labels)[shuffled_indices]

# Split the dataset into train, validation, and test sets
num_images = len(image_filenames)
train_split = int(num_images * 0.6)
val_split = int(num_images * 0.2)

train_image_filenames = image_filenames[:train_split]
train_labels = labels[:train_split]

val_image_filenames = image_filenames[train_split:train_split+val_split]
val_labels = labels[train_split:train_split+val_split]

test_image_filenames = image_filenames[train_split+val_split:]
test_labels = labels[train_split+val_split:]

# Create a LabelEncoder object and fit it to the labels
label_encoder = LabelEncoder()
label_encoder.fit(labels)

# Fit the label encoder on the entire dataset
all_labels = np.concatenate((train_labels, val_labels, test_labels))
label_encoder.fit(all_labels)

# Transform the training, validation, and test labels to integer-encoded format
train_labels = label_encoder.transform(train_labels)
val_labels = label_encoder.transform(val_labels)
test_labels = label_encoder.transform(test_labels)

# Convert the integer-encoded labels to one-hot encoded vectors
num_classes = len(label_encoder.classes_)
train_labels = to_categorical(train_labels, num_classes)
val_labels = to_categorical(val_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

# Load the images into numpy arrays


def load_image(image_filename):
    with Image.open(image_filename) as image:
        image = image.resize((224, 224))  # Resize the images to a consistent size
        image = np.array(image)  # pixel values not normalised
    return image


train_images = np.array([preprocess_input(load_image(filename)) for filename in train_image_filenames])
val_images = np.array([preprocess_input(load_image(filename)) for filename in val_image_filenames])
test_images = np.array([preprocess_input(load_image(filename)) for filename in test_image_filenames])

# Load the MobileNetV2 model without the top layers and set the input shape
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Extract features from the images using the base model
train_features = base_model.predict(train_images)
val_features = base_model.predict(val_images)
test_features = base_model.predict(test_images)

# Flatten the features
train_features = train_features.reshape(train_features.shape[0], -1)
val_features = val_features.reshape(val_features.shape[0], -1)
test_features = test_features.reshape(test_features.shape[0], -1)

# Train a classifier on the extracted features
classifier = Sequential()
classifier.add(Dense(64, activation='relu', input_dim=train_features.shape[1]))
classifier.add(Dropout(0.5))
classifier.add(Dense(2, activation='softmax'))

classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

classifier.fit(train_features, train_labels, epochs=10, batch_size=32, validation_data=(val_features, val_labels))

# Evaluate the classifier on the test set
test_loss, test_acc = classifier.evaluate(test_features, test_labels)

print('Test accuracy:', test_acc)

# Calculate performance metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Get the predicted labels
y_true = np.argmax(test_labels, axis=1)
y_pred = np.argmax(classifier.predict(test_features), axis=1)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

report = classification_report(y_true, y_pred, target_names=[str(cls) for cls in label_encoder.classes_])
print(report)

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

# Save the model
classifier.save('dataset/models/citrus_classifier.h5')

print('Number of healthy images:', sum(labels == 'healthy'))
print('Number of defective images:', sum(labels == 'defective'))

