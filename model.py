import numpy as np
import glob
import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization


print(os.listdir("images/"))

SIZE = 128

train_images = []
train_labels = []
for directory_path in glob.glob("images/train/*"):
    label = directory_path.split("\\")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_images.append(img)
        train_labels.append(label)

train_images = np.array(train_images)
train_labels = np.array(train_labels)

test_images = []
test_labels = []
for directory_path in glob.glob("images/test/*"):
    lbl = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        test_images.append(img)
        test_labels.append(lbl)

test_images = np.array(test_images)
test_labels = np.array(test_labels)

le = preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)

x_train, y_train, x_test, y_test = train_images, train_labels_encoded, test_images, test_labels_encoded

x_train, x_test = x_train / 255.0, x_test / 255.0

activation = 'sigmoid'

feature_extractor = Sequential()
feature_extractor.add(Conv2D(32, 3, activation=activation,
                             padding='same', input_shape=(SIZE, SIZE, 3)))
feature_extractor.add(BatchNormalization())

feature_extractor.add(Conv2D(32, 3, activation=activation,
                             padding='same', kernel_initializer='he_uniform'))
feature_extractor.add(BatchNormalization())
feature_extractor.add(MaxPooling2D())

feature_extractor.add(Conv2D(64, 3, activation=activation,
                             padding='same', kernel_initializer='he_uniform'))
feature_extractor.add(BatchNormalization())

feature_extractor.add(Conv2D(64, 3, activation=activation,
                             padding='same', kernel_initializer='he_uniform'))
feature_extractor.add(BatchNormalization())
feature_extractor.add(MaxPooling2D())

feature_extractor.add(Flatten())

X_for_RF = feature_extractor.predict(x_train)  # This is out X input to RF
RF_model = RandomForestClassifier(n_estimators=50, random_state=42)

# Train the model on training data
RF_model.fit(X_for_RF, y_train)  # For sklearn no one hot encoding

# Send test data through same feature extractor process
X_test_feature = feature_extractor.predict(x_test)
# Now predict using the trained RF model.
prediction_RF = RF_model.predict(X_test_feature)
# Inverse le transform to get original label back.
prediction_RF = le.inverse_transform(prediction_RF)

print("Accuracy = ", metrics.accuracy_score(test_labels, prediction_RF))

# Confusion Matrix - verify accuracy of each class
cm = confusion_matrix(test_labels, prediction_RF)
# print(cm)
sns.heatmap(cm, annot=True)

# Check results on a few select images
# n=5 #dog park. RF works better than CNN
n = 9  # Select the index of image to be loaded for testing
img = x_test[n]
plt.imshow(img)
# Expand dims so the input is (num images, x, y, c)
input_img = np.expand_dims(img, axis=0)
input_img_features = feature_extractor.predict(input_img)
prediction_RF = RF_model.predict(input_img_features)[0]
# Reverse the label encoder to original name
prediction_RF = le.inverse_transform([prediction_RF])
print("The prediction for this image is: ", prediction_RF)
print("The actual label for this image is: ", test_labels[n])
