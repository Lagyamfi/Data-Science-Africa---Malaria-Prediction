"""This file is for patches classification (step 1).

Sample predictive model.
You must supply at least 4 methods:
- fit: trains the model.
- predict: uses the model to perform predictions.
- save: saves the model.
- load: reloads the model.
"""
import numpy as np   # We recommend to use numpy arrays
from sklearn.base import BaseEstimator

#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import models, layers
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class model(BaseEstimator):
    """Main class for Classification problem."""

    def __init__(self):
        """Init method.

        We define here a simple (shallow) CNN.
        """
        self.num_train_samples = 0
        self.num_feat = 1
        self.num_labels = 1
        self.is_trained = False

        self.model = models.Sequential()
        self.model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(40, 40, 3)))
        self.model.add(layers.MaxPooling2D((2,2)))
        self.model.add(layers.Conv2D(64, (3,3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2,2)))
        self.model.add(layers.Conv2D(128, (3,3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2,2)))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dropout(0.6))
        self.model.add(layers.Dense(256, activation='relu'))
        self.model.add(layers.Dense(1, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy',
                           optimizer=optimizers.RMSprop(lr=2e-3),
                           metrics=['accuracy'])
    def fit(self, X, y, validation_data=None, epochs=40):
        """Fit method.

        This function should train the model parameters.
        Args:
            X: Training data matrix of dim num_train_samples * num_feat.
               An image has the following shape (40, 40, 3) then 4800 features.
            y: Training label matrix of dim num_train_samples.
        Both inputs are numpy arrays.
        """
        self.num_train_samples = X.shape[0]
        X = X.reshape((self.num_train_samples, 40, 40, 3))
        train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True,)
        train_generator = train_datagen.flow(X, y=y, batch_size=20)

        self.model.fit_generator(train_generator,
                              steps_per_epoch= self.num_train_samples / 20,
                              epochs=epochs,
                                use_multiprocessing=True,
                                workers=4)
        self.is_trained = True

    def predict(self, X):
        """Predict method.

        Args:
            X: Training data matrix of dim num_train_samples * num_feat.
               An image has the following shape (40, 40, 3) then 4800 features.
        This function should provide predictions of labels on (test) data.
        Make sure that the predicted values are in the correct format for the
        scoring metric. For example, binary classification problems often
        expect predictions in the form of a discriminant value (if the area
        under the ROC curve it the metric) rather that predictions of the class
        labels themselves. For multi-class or multi-labels problems, class
        probabilities are often expected if the metric is cross-entropy.
        Scikit-learn also has a function predict-proba, we do not require it.
        The function predict eventually can return probabilities.
        """
        num_test_samples = X.shape[0]
        X = X.reshape((num_test_samples, 40, 40, 3))
        return self.model.predict_proba(X /255.0)
