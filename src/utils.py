import tensorflow as tf
import matplotlib.pyplot as plt
import os
import datetime
import random


def create_label_mask(label_mask):
    label_mask = tf.argmax(label_mask, axis=-1)
    label_mask = label_mask[..., tf.newaxis]
    return label_mask


def display_image(display_list, epoch):
    plt.figure(num='Epoch ' + str(epoch), figsize=(30, 30))
    title = ['Image', 'Mask', "Predicted Mask"]
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.pause(0.001)


def get_images(path_list, perc=1):
    array = []
    if not isinstance(path_list, list):
        path_list = [path_list]
    for path in path_list:
        for r, d, images in os.walk(path):
            tot = int(perc * len(images))
            saved = 0
            random.shuffle(images)
            for img in images:
                saved += 1
                array.append(os.path.join(r, img))
                if saved > tot:
                    break
    return array


def tf_record_count(tf_record_path):
    return sum(1 for _ in tf.data.TFRecordDataset(tf_record_path))


def get_mask_paths(training_paths):
    mask_paths = []
    for path in training_paths:
        mask_path = path.replace('normal', 'label')
        mask_paths.append(mask_path)
    return mask_paths


def read_pixel_frequency(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if 'BLACK' in line:
                perc_black = line.split('=')[1]
            if 'WHITE' in line:
                perc_white = line.split('=')[1]
    return float(perc_black), float(perc_white)


def get_val_metric(metric_name, val_metrics):
    for m in val_metrics:
        if metric_name in m:
            return m


def plot_metrics(model_history, epochs, save_path):
    l_epochs = epochs
    epochs = range(epochs)
    train_metrics = []
    val_metrics = []
    for key in model_history:
        if 'train' in key:
            train_metrics.append(key)
        else:
            val_metrics.append(key)
    for metric in train_metrics:
        metric_name = metric.split('_')[1]
        mv = model_history[get_val_metric(metric_name, val_metrics)]
        mt = model_history[metric]
        labelt = 'Training'
        labelv = 'Validation'
        plt.figure()
        colorv = 'red'
        colort = 'blue'
        plt.plot(epochs, mt, color=colort, linestyle='-', label=labelt + ' ' + metric_name)
        plt.plot(epochs, mv, color=colorv, linestyle='-', label=labelv + ' ' + metric_name)
        plt.title(metric_name)
        plt.xlabel('Epoch')
        plt.ylabel(metric_name + ' value')
        plt.ylim([0, 1])
        plt.legend()
        plt.savefig(os.path.join(save_path, metric_name + '_' + str(l_epochs) + '.jpg'))
        plt.pause(0.001)
        plt.close()


def create_folder_and_save_path(dir_path, model_name, split=True):
    folder_name = model_name + "_" + str(datetime.datetime.now()).replace(':', '_')
    folder_path = dir_path + folder_name
    os.mkdir(folder_path)
    if split:
        os.mkdir(folder_path + '/Training')
        os.mkdir(folder_path + '/Validation')
    print("CREATE ", folder_path)
    return folder_path


def plot(dir_path, model_name, model_history, epochs,):
    save_path = create_folder_and_save_path(dir_path, model_name, split=False)
    plot_metrics(model_history, epochs, save_path)


def save_test(image, logit, path, index):
    logit = tf.squeeze(create_label_mask(logit), axis=-1)
    plt.figure(num='Number ' + str(index), figsize=(30, 30))
    plt.subplot(1, 2, 1)
    plt.title('Image')
    plt.imshow(image)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title('Prediction')
    plt.imshow(logit)
    plt.axis('off')
    plt.savefig(os.path.join(path, 'test_' + str(index) + '.png'))
    plt.close()


def save_validation(image, mask, logit, path, index):
    logit = tf.squeeze(create_label_mask(logit), axis=-1)
    plt.figure(num='Number ' + str(index), figsize=(30, 30))
    plt.subplot(1, 3, 1)
    plt.title('Image')
    plt.imshow(image)
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.title('True Mask')
    plt.imshow(tf.keras.preprocessing.image.array_to_img(mask))
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.title('Prediction')
    plt.imshow(logit)
    plt.axis('off')
    plt.savefig(os.path.join(path, 'prediction_' + str(index) + '.png'))
    plt.close()
    #img = tf.keras.preprocessing.image.array_to_img(save_image)
    #img.save(os.path.join(path, 'prediction_' + index + '.png'))

