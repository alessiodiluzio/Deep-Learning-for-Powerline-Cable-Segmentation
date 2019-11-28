import tensorflow as tf
from src.unet import Encoder
from src.unet import Decoder
from src.metrics import precision_recall, compute_accuracy, compute_f1score
from src.utils import read_pixel_frequency
from src.utils import display_image, create_label_mask
import os


class UnetModel(tf.keras.Model):

    def __init__(self, name, device, checkpoint_dir):
        super(UnetModel, self).__init__(name="UnetModel")
        self._device = device
        self._checkpoint_dir = os.path.join(checkpoint_dir, name)
        self._is_built = False
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.history = {}
        self.BLACK_PERCENTUAL, self.WHITE_PERCENTUAL = read_pixel_frequency('file/WHITE_BLACK_PERCENTUAL.txt')

    @property
    def device(self):
        return self._device

    @property
    def checkpoint_dir(self):
        return self._checkpoint_dir

    @property
    def saver(self):
        return tf.compat.v1.train.Saver(self.variables)

    @property
    def is_built(self):
        return self._is_built

    @is_built.setter
    def is_built(self, value):
        if isinstance(value, bool) is False:
            raise ValueError("is_built must be a boolean value")
        self._is_built = value

    def forward(self, *args, **kwargs):
        with tf.device(self.device):
            output = self.call(*args, **kwargs)
        self.is_built = True
        return output

    def call(self, image, training=False):
        input_tensor = image  # tf.concat([imgs_1, imgs_2], axis=-1)
        x = self.encoder(input_tensor, training=training)
        skip_connections = self.encoder.skip_connections
        x = self.decoder(x, training=training, skip_connections=skip_connections)
        return x

    def compute_cross_entropy(self, image, one_hot_label, training=True):
        logits = self.forward(image, training=training)
        cross_entropy = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_label, logits=logits)
        if training:
            class_weights = tf.constant([[[[1/self.BLACK_PERCENTUAL, 1/self.WHITE_PERCENTUAL]]]])
            weights = tf.reduce_sum(class_weights * one_hot_label, axis=-1)
            weighted_loss = tf.reduce_mean(cross_entropy * weights)
            return weighted_loss
        return cross_entropy

    def backward(self, image, one_hot_label):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                loss = self.compute_cross_entropy(image, one_hot_label, training=True)
            gradients = tape.gradient(loss, self.variables)
            grad_vars = zip(gradients, self.variables)
            return loss, grad_vars

    def load_variables(self, sess=None):
        if self.is_built is False:
            # Run the model once to initialize variables
            dummy_image = tf.zeros((1, 256, 256, 3))
            self.forward(dummy_image, training=False)
            self.saver.restore(sess=sess, save_path=self.checkpoint_dir)

    def save_variables(self, sess=None):
        if self.is_built:
            self.saver.save(sess=sess, save_path=self.checkpoint_dir)
        else:
            raise RuntimeError("You must build the model before save model's variables.")

    def train(self, train_dataset, val_dataset, optimizer, train_steps, val_steps, epochs=50):

        if tf.executing_eagerly() is False:
            raise RuntimeError("train method must be run only with eager execution.")
        global_step = tf.compat.v1.train.get_or_create_global_step()
        # best_f1score = 0
        best_acc = 0
        # Initialize dictionary to store the history
        self.history = {'train_loss': [], 'val_loss': [], 'train_f1score': [], 'val_f1score': [], 'train_acc': [],
                        'val_acc': []}

        for i in range(epochs):
            print("Epoch: {}/{}".format(i + 1, epochs))

            train_loss = tf.metrics.Mean('train_loss')
            train_f1score = tf.metrics.Mean('train_f1')
            train_accuracy = tf.metrics.Mean('train_acc')

            train_progbar = tf.keras.utils.Progbar(train_steps)

            for b, (image, mask) in enumerate(train_dataset):
                mask = tf.squeeze(mask, axis=-1)
                one_hot_labels = tf.one_hot(indices=mask, depth=2, dtype=tf.float32)
                mask = tf.cast(mask, tf.float32)
                loss, grads = self.backward(image, one_hot_labels)
                optimizer.apply_gradients(grads, global_step)
                logits = self.forward(image, training=False)
                precision, recall = precision_recall(logits, mask)
                f1score = compute_f1score(precision, recall)
                accuracy = compute_accuracy(logits, mask)
                train_loss(loss)
                train_f1score(f1score)
                train_accuracy(accuracy)
                metrics = [('loss', loss), ("f1", f1score), ("accuracy", accuracy)]
                train_progbar.update(b + 1, metrics)

            val_loss = tf.metrics.Mean('val_loss')
            val_f1score = tf.metrics.Mean('val_f1')
            val_accuracy = tf.metrics.Mean('val_acc')

            val_progbar = tf.keras.utils.Progbar(val_steps)
            print("\nVALIDATION")
            for b, (image, mask) in enumerate(val_dataset):
                mask = tf.squeeze(mask, axis=-1)
                one_hot_labels = tf.one_hot(indices=mask, depth=2, dtype=tf.float32)
                mask = tf.cast(mask, tf.float32)

                loss = self.compute_cross_entropy(image, one_hot_labels, training=False)
                logits = self.forward(image, training=False)

                precision, recall = precision_recall(logits, mask)
                f1score = compute_f1score(precision, recall)
                accuracy = compute_accuracy(logits, mask)
                val_loss(loss)
                val_f1score(f1score)
                val_accuracy(accuracy)
                metrics = [('val_loss', loss), ("val_f1", f1score), ("val_acc", accuracy)]
                val_progbar.update(b + 1, metrics)
            i = 0
            for b, (image, mask) in enumerate(val_dataset):
                if i >= 2:
                    break
                logits = self.forward(image, training=False)
                display_image([image[0], mask[0], create_label_mask(logits[0])])
                i += 1

            self.history['train_loss'].append(train_loss.result().numpy())
            self.history['val_loss'].append(val_loss.result().numpy())
            self.history['train_acc'].append(train_accuracy.result().numpy())

            self.history['train_f1score'].append(train_f1score.result().numpy())
            self.history['val_f1score'].append(val_f1score.result().numpy())
            self.history['val_acc'].append(val_accuracy.result().numpy())

            if self.history['val_acc'][-1] >= best_acc:
                self.save_variables()
                print("Model saved. f1: {} --> {}".format(best_acc, self.history['val_acc'][-1]))
                best_acc = self.history['val_acc'][-1]

    def evaluate(self, test_dataset, steps):
        if tf.executing_eagerly() is False:
            raise RuntimeError("evaluate method must be run only with eager execution.")
        self.load_variables()
        precision_mean = tf.metrics.Mean('precision')
        recall_mean = tf.metrics.Mean('recall')
        accuracy_mean = tf.metrics.Mean('accuracy')
        f1score_mean = tf.metrics.Mean('f1score')
        progbar = tf.keras.utils.Progbar(steps)

        for b, (image, mask) in enumerate(test_dataset):
            # labels = tf.cast(tf.squeeze(labels, axis=-1), tf.float32)
            logits = self.forward(image, mask, training=False)

            precision, recall = self.precision_recall(logits, mask)
            accuracy = self.accuracy(logits, mask)
            f1score = self.f1score(precision, recall)

            precision_mean(precision)
            recall_mean(recall)
            accuracy_mean(accuracy)
            f1score_mean(f1score)
            progbar.update(b + 1)

        print("EVALUATE {}".format(self.name))
        print("precision: {}".format(precision_mean.result().numpy()))
        print("recall: {}".format(recall_mean.result().numpy()))
        print("accuracy: {}".format(accuracy_mean.result().numpy()))
        print("f1score: {}".format(f1score_mean.result().numpy()))
