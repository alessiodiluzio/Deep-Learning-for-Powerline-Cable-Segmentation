import tensorflow as tf
from src.utils import get_images, get_mask_paths
import random

IMAGE_PATH = "file/input/normal"

images = get_images(IMAGE_PATH)


class TFRecordEncoder(object):

    def __init__(self, images_paths, train_record_file, validation_record_file, perc_train):
        self.train_record_file = train_record_file
        self.validation_record_file = validation_record_file
        self.perc_train = perc_train
        self.dataset_size = len(images_paths)
        self.train_images_paths, self.validation_images_paths = self.get_train_validation_split(images_paths)
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
        image_shape = tf.image.decode_jpeg(image_string).shape
        feature = {
            'height': self._int64_feature(image_shape[0]),
            'width': self._int64_feature(image_shape[1]),
            'depth': self._int64_feature(image_shape[2]),
            'image_raw': self._bytes_feature(image_string),
            'mask_raw': self._bytes_feature(mask_string)
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def tf_record_writer(self):
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
        return


record_encoder = TFRecordEncoder(images, 'TFRecords/training.record', 'TFRecords/validation.record', 0.8)
record_encoder.tf_record_writer()

"""""
class TFRecordDecoder(object):

    def __init__(self, records_path):
        self.records_path = records_path
        self.image_feature_description = {
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'depth': tf.io.FixedLenFeature([], tf.int64),
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'mask_raw': tf.io.FixedLenFeature([], tf.string),

        }

    def _parse_image_function(self, example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, self.image_feature_description)

    def get_parsed_image_dataset(self):
        raw_image_dataset = tf.data.TFRecordDataset(self.records_path)
        return raw_image_dataset.map(self._parse_image_function)

    
        i = 1;
        for image_features in parsed_image_dataset:
            if i > 3:
                return
            image_raw = image_features['image_raw'].numpy()
            image = tf.io.decode_png(image_raw, 3)
            mask_raw = image_features['mask_raw'].numpy()
            mask = tf.io.decode_png(mask_raw, 3)
            display_image([image, mask])
            i += 1
        

    def test_display(self):
        dataset = self.test()
        for image, mask in dataset:
            display(Image(data=image))
"""""
