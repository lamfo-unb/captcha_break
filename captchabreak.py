from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import cv2
from imutils import paths
import os.path

'''
Train/Predict code
'''

letter_folder = 'LettersCaptchas'

# creating empty lists for storing image data and labels
data = []
labels = []
for image in paths.list_images(letter_folder):
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (40, 40))

    # adding a 3rd dimension to the image
    img = np.expand_dims(img, axis=2)

    # grabing the name of the letter based on the folder it is present in
    label = image.split(os.path.sep)[-2]

    # appending to the empty lists
    data.append(img)
    labels.append(label)

# converting data and labels to np array
data = np.array(data, dtype="float")
labels = np.array(labels)

# scaling the values of data between 0 and 1
data = data / 255.0

# Split the training data into separate train and test sets
(train_x, val_x, train_y, val_y) = train_test_split(data, labels, test_size=0.3, random_state=13)

# one hot encoding
lb = LabelBinarizer().fit(train_y)
train_y = lb.transform(train_y)
val_y = lb.transform(val_y)

# building model
model = Sequential()
model.add(Conv2D(40, (5, 5), padding="same", input_shape=(40, 40, 1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(100, (5, 5), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(26, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# using early stoping for avoiding overfitting
estop = EarlyStopping(patience=20, mode='min', min_delta=0.001, monitor='val_loss')

model.fit(train_x, train_y, validation_data=(val_x, val_y), batch_size=20, epochs=500, verbose=1, callbacks=[estop])

# Load the image and convert it to grayscale
image = cv2.imread('captcha.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Add some extra padding around the image
gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 10, 50)

# threshold the image
thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV, cv2.THRESH_OTSU)[1]

# find the contours
contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

letter_image_regions = []

# Now we can loop through each of the contours and extract the letter
for contour in contours:
    # Get the rectangle that contains the contour
    (x, y, w, h) = cv2.boundingRect(contour)

    # checking if any countour is too wide
    # if countour is too wide then there could be two letters joined together or are very close to each other
    if w / h > 5:
        # Split it in half into two letter regions
        half_width = int(w / 2)
        letter_image_regions.append((x, y, half_width, h))
        letter_image_regions.append((x + half_width, y, half_width, h))
    else:
        letter_image_regions.append((x, y, w, h))

# Sort the detected letter images based on the x coordinate to make sure
# we get them from left-to-right so that we match the right image with the right letter

letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

# Create an output image and a list to hold our predicted letters
output = cv2.merge([gray] * 3)
predictions = []

# Creating an empty list for storing predicted letters
predictions = []

# Save out each letter as a single image
for letter_bounding_box in letter_image_regions:
    # Grab the coordinates of the letter in the image
    x, y, w, h = letter_bounding_box

    # Extract the letter from the original image with a 2-pixel margin around the edge
    letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]

    letter_image = cv2.resize(letter_image, (40, 40))

    # Turn the single image into a 4d list of images
    letter_image = np.expand_dims(letter_image, axis=2)
    letter_image = np.expand_dims(letter_image, axis=0)

    # making prediction
    pred = model.predict(letter_image)

    # Convert the one-hot-encoded prediction back to a normal letter
    letter = lb.inverse_transform(pred)[0]
    # TODO: o modelo esta vendo I aonde nao tem, ele so quebra captcha que nao tem a letra I
    # TODO: adicionar mais captchas para treinamento e separar palavras nas pastas
    if letter != 'I':
        predictions.append(letter)

# Print the captcha's text
captcha_text = "".join(predictions)
print("\nCAPTCHA text is: \n{}".format(captcha_text))
