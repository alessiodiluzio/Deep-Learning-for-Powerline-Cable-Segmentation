import tensorflow as tf
from src.utils import get_images

IMAGE_PATH = "C:\\Users\\aless\\OneDrive\\Desktop\\DATASET\\cable_rect_smaller\\normal"
LABEL_PATH = "C:\\Users\\aless\\OneDrive\\Desktop\\DATASET\\cable_rect_smaller\\label"

images = get_images(IMAGE_PATH)
labels = get_images(LABEL_PATH)


class TFRecordEncoder(object):

    def __init__(self, images_paths, mask_paths):
        self.images_paths = images_paths
        self.mask_paths = mask_paths

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
        record_file = 'images.tfrecords'
        with tf.io.TFRecordWriter(record_file) as writer:
            for image, mask in zip(self.images_paths, self.mask_paths):
                tf_example = self._tf_record_example(image, mask)
                writer.write(tf_example.SerializeToString())

        return

    def get_some(self):
        img_path = self.images_paths[0]
        lbl_path = self.mask_paths[0]
        for line in str(self._tf_record_example(img_path, lbl_path)).split('\n')[:15]:
            print(line)
        print('...')


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
