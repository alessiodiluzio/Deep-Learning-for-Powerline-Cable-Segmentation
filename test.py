import tensorflow as tf
import sys
from src.dataset import TestDataLoader
from src.utils import tf_record_count, display_image
from src.model import CableModel



def main(_):
    device = 'cpu:0'
    print(tf.test.is_gpu_available())
    if tf.test.is_gpu_available(cuda_only=True):
        device = 'gpu:0'
    net = CableModel('Unet', device, "checkpoint/150_epoch_unet_2")

    test_batch_size = 1

    test_dataset = TestDataLoader(image_size=[[1024, 1024], [1024, 1024]],
                                  tf_records_path=['TFRecords/real_video/real_video.record'])
    test_dataset = test_dataset.data_batch(batch_size=test_batch_size, augmentation=False)

    for image in test_dataset.take(3):
        display_image([image[0]], 0)

    test_steps = int(tf_record_count(['TFRecords/real_video/real_video.record'])/test_batch_size)
    net.load_variables()
    net.test(test_dataset, test_steps)
    sys.exit(0)


if __name__ == '__main__':
    tf.compat.v1.app.run()