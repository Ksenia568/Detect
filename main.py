import numpy as np
import telebot
import tensorflow as tf
from tensorflow import keras

img_height = 180
img_width = 180

class_names = ['Яблоко', 'Гнилое яблоко', 'Банан', 'Гнилой банан', 'Клубнику', 'Гнилую клубнику']

model = keras.models.load_model('templates/flowers_wow_model.h5')

bot = telebot.TeleBot('6007554429:AAHUL8zX2WmO7SGtHJLtjka3-xbjBSG24F4')


@bot.message_handler(commands=['start'])
def handle_start(message):
    bot.reply_to(message, "Привет! Я могу определить название цветка по фотографии. "
                          "А именно ромашку, одуванчик, розу, подсолнух и тюльпан. Прикрепите фото*")
    bot.user_data = {}


@bot.message_handler(content_types=['text'])
def bot_message(message):
    if message.chat.type == 'private':
        bot.send_message(message.chat.id, "Отправьте картинку ;)")


@bot.message_handler(content_types=['photo'])
def photo(message):
    fileID = message.photo[-1].file_id
    file_info = bot.get_file(fileID)
    downloaded_file = bot.download_file(file_info.file_path)
    with open("image.jpg", 'wb') as new_file:
        new_file.write(downloaded_file)
    img = tf.keras.utils.load_img(
        "image.jpg", target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    flower = class_names[np.argmax(score)]
    percent = round(100 * np.max(score), 2)

    bot.reply_to(message, f"Это изображение похоже на {flower} с вероятностью {percent}%.")


bot.polling()
