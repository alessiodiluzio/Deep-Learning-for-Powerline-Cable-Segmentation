import tensorflow as tf
from src.utils import get_images, get_mask_paths
from script.pixel_frequency import write_pixel_frequency
import random


class TFRecordEncoder(object):

    def __init__(self, images_path, train_record_file, validation_record_file, perc_train, perc_dataset=1):
        self.train_images_paths = get_images(images_path, perc_dataset)
        print(self.train_images_paths)
        random.shuffle(self.train_images_paths)
        self.train_record_file = train_record_file
        self.validation_record_file = validation_record_file
        self.perc_train = perc_train
        self.dataset_size = len(self.train_images_paths)
        self.train_images_paths, self.validation_images_paths = self.get_train_validation_split(self.train_images_paths)
        self.train_mask_paths = get_mask_paths(self.train_images_paths)
        self.validation_mask_paths = get_mask_paths(self.validation_images_paths)

    def get_train_validation_split(self, images_path):
        training = []
        validation = []
        random.shuffle(images_path)
        train_size = int(self.perc_train * self.dataset_size)
        for i in range(0, train_size):
            training.append(images_path[i])
        for i in range(train_size, self.dataset_size):
            validation.append(images_path[i])
        return training, validation

    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _tf_record_example(self, image_path, mask_path):
        image_string = open(image_path, 'rb').read()
        mask_string = open(mask_path, 'rb').read()
        image_shape = tf.image.decode_png(image_string).shape
        feature = {
            'height': self._int64_feature(image_shape[0]),
            'width': self._int64_feature(image_shape[1]),
            'depth': self._int64_feature(image_shape[2]),
            'image_raw': self._bytes_feature(image_string),
            'mask_raw': self._bytes_feature(mask_string)
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def tf_record_writer(self, change_percentual=False):
        train_record_file = self.train_record_file
        validation_record_file = self.validation_record_file
        with tf.io.TFRecordWriter(train_record_file) as writer:
            for image, mask in zip(self.train_images_paths, self.train_mask_paths):
                tf_example = self._tf_record_example(image, mask)
                writer.write(tf_example.SerializeToString())
        with tf.io.TFRecordWriter(validation_record_file) as writer:
            for image, mask in zip(self.validation_images_paths, self.validation_mask_paths):
                tf_example = self._tf_record_example(image, mask)
                writer.write(tf_example.SerializeToString())
        if change_percentual:
            write_pixel_frequency('../file/WHITE_BLACK_PERCENTUAL.txt', self.train_mask_paths)
        return


class TFRecordEncoderTest(object):

    def __init__(self, images_path, test_record_file):
        self.test_images_paths = get_images(images_path)
        self.test_record_file = test_record_file
        self.dataset_size = len(self.test_images_paths)

    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _tf_record_example(self, image_path):
        image_string = tf.io.read_file(image_path)
        image_shape = tf.image.decode_png(image_string).shape
        feature = {
            'height': self._int64_feature(image_shape[0]),
            'width': self._int64_feature(image_shape[1]),
            'depth': self._int64_feature(image_shape[2]),
            'image_raw': self._bytes_feature(image_string),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def tf_record_writer(self):
        test_record_file = self.test_record_file
        with tf.io.TFRecordWriter(test_record_file) as writer:
            for image in self.test_images_paths:
                tf_example = self._tf_record_example(image)
                writer.write(tf_example.SerializeToString())
        return




#record_encoder = TFRecordEncoder("../file/input/large/normal", '../TFRecords/large/training_large.record',
                               #  '../TFRecords/large/validation_large.record', 0.8, 1)
#record_encoder.tf_record_writer(change_percentual=True)
test_encoder = TFRecordEncoderTest("../file/input/real", '../TFRecords/real_video/real.record')
test_encoder.tf_record_writer()
