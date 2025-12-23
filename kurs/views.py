from django.shortcuts import render
import numpy as np
from tensorflow import keras
import tensorflow as tf
import os.path
import os
import shutil


def recognition(request):
    img_height = 180
    img_width = 180
    class_names = ['Свежее яблоко', 'Гнилое яблоко', 'Свежий банан', 'Гнилой банан', 'Свежую клубнику', 'Гнилую клубнику']

    os.getcwd()
    collection = "media/images/"

    for i, filename in enumerate(os.listdir(collection)):
        os.rename(collection + filename, collection + str(i) + ".jpg")

    if os.path.isfile(collection + "0.jpg"):
        shutil.copy2(collection + "0.jpg", "templates")

    for i, filename in enumerate(os.listdir(collection)):
        os.remove(collection + filename)

    model = keras.models.load_model("templates/fruit.h5")
    if os.path.isfile("templates/0.jpg"):
        img = tf.keras.utils.load_img("templates/0.jpg", target_size=(img_height, img_width))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch.

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        image_01 = class_names[np.argmax(score)]
        percent = round(100 * np.max(score), 2)
        return render(request, "main/home.html", {"image_01": image_01, 'percent': percent})
    else:
        return render(request, "main/home.html", {"image_01": "???", 'percent': "???"})
