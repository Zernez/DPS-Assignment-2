# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np


def Train_model():
    # build model
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        # tf.keras.layers.Flatten(input_shape(11,100)),
        tf.keras.layers.Dense(128, activation='relu'),
        #tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=10)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    print('\nTest accuracy:', test_acc)

    return model

def Predict_model(model, test_data, categories):
    probability_model = tf.keras.Sequential([model,
                                             tf.keras.layers.Softmax()])
    predictions = probability_model.predict(test_data)

    predicted_labels = []
    predicted_cat = []
    for pred in predictions:
        lab = np.argmax(pred)
        predicted_labels.append(lab)
        predicted_cat.append(categories[lab])

    print(predicted_cat)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("TensorFlow version:", tf.__version__)

    #load data
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (tra_images, tra_labels), (t_images, t_labels) = fashion_mnist.load_data()
    train_images, test_images,train_labels, test_labels = train_test_split(tra_images, tra_labels, test_size=0.2)

    categories = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    #build model
    model = Train_model()
    #predict
    prediction = Predict_model(model, test_images, categories)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
