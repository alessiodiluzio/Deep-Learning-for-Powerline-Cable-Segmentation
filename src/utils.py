import tensorflow as tf
import matplotlib.pyplot as plt
import os


def create_label_mask(label_mask):
    label_mask = tf.argmax(label_mask, axis=-1)
    label_mask = label_mask[..., tf.newaxis]
    return label_mask


def display_image(display_list):
    plt.figure(figsize=(30, 30))
    title = ['Image', 'Mask', "Predicted Mask"]
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def get_images(path):
    array = []
    for r, d, images in os.walk(path):
        for img in images:
            array.append(os.path.join(r, img))
    return array
