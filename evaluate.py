import tensorflow as tf
import sys
from src.dataset import DataLoader
from src.utils import tf_record_count
from src.model import CableModel


def main(_):
    device = 'cpu:0'
    print(tf.test.is_gpu_available())
    if tf.test.is_gpu_available(cuda_only=True):
        device = 'gpu:0'
    net = CableModel('Unet', device, "checkpoint")
    validation_batch_size = 10
    validation_dataset = DataLoader(image_size=[[128, 128], [128, 128]],
                                    tf_records_path=['TFRecords/small/validation_small.record'])
    validation_dataset = validation_dataset.data_batch(batch_size=validation_batch_size, augmentation=False)
    validation_steps = int(tf_record_count(['TFRecords/small/validation_small.record'])/validation_batch_size)
    net.evaluate(validation_dataset, validation_steps)
    sys.exit(0)


if __name__ == '__main__':
    tf.compat.v1.app.run()