import tensorflow as tf
import sys
from src.dataset import TestDataLoader
from src.utils import tf_record_count
from src.model import CableModel


def main(_):
    device = 'cpu:0'
    print(tf.test.is_gpu_available())
    if tf.test.is_gpu_available(cuda_only=True):
        device = 'gpu:0'
    net = CableModel('Unet', device, "checkpoint")

    test_batch_size = 10

    test_dataset = TestDataLoader(image_size=[[128, 128], [128, 128]],
                                  tf_records_path=['TFRecords/real/real.record'])
    test_dataset = test_dataset.data_batch(batch_size=test_batch_size, augmentation=False)

    test_steps = int(tf_record_count(['TFRecords/real/real.record'])/test_batch_size)
    net.load_variables()
    net.test(test_dataset, test_steps)
    sys.exit(0)


if __name__ == '__main__':
    tf.compat.v1.app.run()