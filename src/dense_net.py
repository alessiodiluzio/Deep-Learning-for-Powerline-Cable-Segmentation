import tensorflow as tf
import os
from .metrics import precision_recall, compute_accuracy, compute_f1score
from .utils import read_pixel_frequency, plot, save_validation, create_folder_and_save_path, save_test, create_label_mask
from IPython.display import clear_output


class Layer(tf.keras.Model):

    def __init__(self, kernel_initializer):
        super(Layer, self).__init__(name='layer')
        self.convolution = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1,
                                                  padding="same", activation=None,
                                                  kernel_initializer=kernel_initializer)
        self.batch_normalization = tf.keras.layers.BatchNormalization(axis=-1)
        self.dropout = tf.keras.layers.Dropout(0.2, noise_shape=None, seed=None)

    def __call__(self, input_tensor, training=False):

        x = self.batch_normalization(input_tensor, training=training)
        x = tf.nn.relu(x)
        x = self.convolution(x)
        x = self.dropout(x)

        return x


class DenseBlock(tf.keras.Model):

    def __init__(self, num_of_layers):
        super(DenseBlock, self).__init__(name='dense_block')
        self.dense_layers = []
        for i in range(num_of_layers):
            self.dense_layers.append(Layer('he_normal'))
        self.concatenate = tf.keras.layers.Concatenate(axis=-1)

    def __call__(self, input_tensor, training=False):
        input = input_tensor
        output_list = []
        for layer in self.dense_layers[:-1]:
            output = layer(input)
            if not isinstance(input, list):
                input = [input]
            if not isinstance(output, list):
                output = [output]
            output_list.append(output)
            maps = input + output
            input = self.concatenate(maps)
        output = self.dense_layers[-1](input, training=training)
        if not isinstance(output, list):
            output = [output]
        output_list.append(output)
        maps = output_list[0]
        for out in output_list[1:]:
            maps += out
        return self.concatenate(maps)


class TransitionDown(tf.keras.Model):

    def __init__(self, num_input_features):
        super(TransitionDown, self).__init__(name='transition_down')
        self.batch_normalization = tf.keras.layers.BatchNormalization(axis=-1)
        self.convolution = tf.keras.layers.Conv2D(filters=num_input_features, kernel_size=1, strides=1,
                                                  padding="same", activation=None)
        self.dropout = tf.keras.layers.Dropout(0.2, noise_shape=None, seed=None)
        self.max_pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))

    def __call__(self, input_tensor, training=False):
        x = self.batch_normalization(input_tensor, training=training)
        x = tf.nn.relu(x)
        x = self.convolution(x)
        x = self.dropout(x)
        x = self.max_pooling(x)
        return x


class TransitionUp(tf.keras.Model):

    def __init__(self, num_input_features):
        super(TransitionUp, self).__init__(name="transition_up")
        self.transpose_convolution = tf.keras.layers.Conv2DTranspose(filters=num_input_features, kernel_size=3,
                                                                     strides=2, padding="same")

    def __call__(self, input_tensor):
        return self.transpose_convolution(input_tensor)


class DenseNet(tf.keras.Model):
    def __init__(self, name, device, checkpoint_dir):
        super(DenseNet, self).__init__(name='DenseNet')
        self._device = device
        self._checkpoint_dir = os.path.join(checkpoint_dir, name)
        self._is_built = False
        self.first_convolution = tf.keras.layers.Conv2D(filters=48, kernel_size=3, strides=1, padding="same",
                                                        kernel_initializer='he_normal')
        self.concatenate = tf.keras.layers.Concatenate(axis=-1)

        # UpSampling Path
        self.skip_maps = []

        self.dense_block_1 = DenseBlock(4)
        self.transition_down_1 = TransitionDown(112)

        self.dense_block_2 = DenseBlock(5)
        self.transition_down_2 = TransitionDown(192)

        self.dense_block_3 = DenseBlock(7)
        self.transition_down_3 = TransitionDown(304)

        self.dense_block_4 = DenseBlock(10)
        self.transition_down_4 = TransitionDown(464)

        self.dense_block_5 = DenseBlock(12)
        self.transition_down_5 = TransitionDown(656)

        self.downsampling_path_dense_block = [self.dense_block_1,
                                              self.dense_block_2,
                                              self.dense_block_3,
                                              self.dense_block_4,
                                              self.dense_block_5]

        self.downsampling_path_transition_down = [self.transition_down_1,
                                                  self.transition_down_2,
                                                  self.transition_down_3,
                                                  self.transition_down_4,
                                                  self.transition_down_5]

        # Bottleneck

        self.bottleneck = DenseBlock(15)

        # DownSampling Path

        self.transition_up_1 = TransitionUp(240)
        self.dense_block_6 = DenseBlock(12)

        self.transition_up_2 = TransitionUp(192)
        self.dense_block_7 = DenseBlock(10)

        self.transition_up_3 = TransitionUp(160)
        self.dense_block_8 = DenseBlock(7)

        self.transition_up_4 = TransitionUp(112)
        self.dense_block_9 = DenseBlock(5)

        self.transition_up_5 = TransitionUp(80)
        self.dense_block_10 = DenseBlock(4)

        self.upsampling_path_dense_block = [self.dense_block_6,
                                            self.dense_block_7,
                                            self.dense_block_8,
                                            self.dense_block_9,
                                            self.dense_block_10]

        self.upsampling_path_transition_up = [self.transition_up_1,
                                              self.transition_up_2,
                                              self.transition_up_3,
                                              self.transition_up_4,
                                              self.transition_up_5]

        self.last_convolution = tf.keras.layers.Conv2D(filters=2, kernel_size=1, strides=1, padding='same',
                                                       activation=None, kernel_initializer="he_normal")

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

    def call(self, image, training=False):

        input_tensor = image

        # ENCODER

        block_input = self.first_convolution(input_tensor)
        for i in range(len(self.downsampling_path_dense_block)):
            block_output = self.downsampling_path_dense_block[i](block_input, training=training)
            if not isinstance(block_input, list):
                block_input = [block_input]
            if not isinstance(block_output, list):
                block_output = [block_output]
            block_output = self.concatenate(block_input + block_output)
            self.skip_maps.append(block_output)
            block_input = self.downsampling_path_transition_down[i](block_output, training=training)

        # BOTTLENECK

        block_output = self.bottleneck(block_input)
        block_input = block_output

        # DECODER

        for i in range(len(self.upsampling_path_dense_block)):
            block_output = self.upsampling_path_transition_up[i](block_input)
            #block_input = block_output
            #if not isinstance(block_output, list):
            #    block_output = [block_output]
            block_output = self.concatenate([self.skip_maps[-1 - i]] + [block_output])
            block_input = self.upsampling_path_dense_block[i](block_output, training=training)

        return self.last_convolution(block_input)

    def forward(self, *args, **kwargs):
        with tf.device(self.device):
            output = self.call(*args, **kwargs)
        self.is_built = True
        return output

    def softmax_cross_entropy(self, labels, logits):
        positive = labels * tf.math.log(tf.nn.softmax(logits))
        #print(positive)
        positive = tf.where(tf.equal(positive, 0), positive, positive)
        return tf.math.reduce_sum(-1 * positive, axis=-1)

    def sigmoid_cross_entropy(self, labels, logits):
        positive = - labels * tf.math.log(tf.nn.sigmoid(logits))
        negative = (1 - labels) * - tf.math.log(tf.nn.sigmoid(1.0 - logits))
        print(positive + negative)
        print(tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
        positive = tf.where(tf.equal(positive, 0), positive, positive)
        return tf.math.reduce_sum(-1 * positive, axis=-1)

    def compute_cross_entropy(self, image, one_hot_label, training=True):
        logits = self.forward(image, training=training)
        cross_entropy = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_label, logits=logits)
        if training:
            class_weights = tf.constant([[[[1.0/self.BLACK_PERCENTUAL, 1.0/self.WHITE_PERCENTUAL]]]])
            weights = tf.reduce_sum(class_weights * one_hot_label, axis=-1)
            weighted_loss = tf.reduce_mean(cross_entropy * weights)
            return weighted_loss
        return cross_entropy

    def class_balanced_loss(self, image, one_hot_label, training=True):
        beta = self.WHITE_PERCENTUAL
        logits = self.forward(image, training=training)
        cross_entropy = self.sigmoid_cross_entropy(one_hot_label, logits)
        cross_entropy2 = self.softmax_cross_entropy(one_hot_label, logits)
        if training:
            class_weights = tf.constant([[[[beta, 1.0 - beta]]]])
            weights = tf.reduce_sum(class_weights * one_hot_label, axis=-1)
            weighted_loss = tf.reduce_mean(cross_entropy * weights)
            return weighted_loss
        return cross_entropy

    def backward(self, image, one_hot_label):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                loss = self.compute_cross_entropy(image, one_hot_label, training=True)
                # loss = self.class_balanced_loss(image, one_hot_label)
            gradients = tape.gradient(loss, self.trainable_variables)
            grad_vars = zip(gradients, self.trainable_variables)
            return loss, grad_vars

    def load_variables(self, sess=None, num_bands=3):
        """"
        Function to restore trained model.
        """

        if self.is_built is False:
            # Run the model once to initialize variables
            image = tf.zeros((1, 128, 128, num_bands))
            self.forward(image,  training=False)
        self.saver.restore(sess=sess, save_path=self.checkpoint_dir)

    def save_variables(self, sess=None):
        """
        Function to save trained model.
        """
        if self.is_built:
            self.saver.save(sess=sess, save_path=self.checkpoint_dir)
        else:
            raise RuntimeError("You must build the model before save model's variables.")

    def train(self, train_dataset, val_dataset, optimizer, train_steps, val_steps, plot_path, epochs=50):

        if tf.executing_eagerly() is False:
            raise RuntimeError("train method must be run only with eager execution.")
        global_step = tf.compat.v1.train.get_or_create_global_step()
        best_f1score = 0

        # Initialize dictionary to store the history
        self.history = {'train_loss': [], 'val_loss': [], 'train_f1score': [], 'val_f1score': [], 'train_acc': [],
                        'val_acc': []}
        save_path = create_folder_and_save_path('file/training_predictions/',  self.name, split=False)
        for i in range(epochs):
            epoca = i
            clear_output()
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
                # loss = self.class_balanced_loss(image, one_hot_labels, training=False)
                logits = self.forward(image, training=False)

                precision, recall = precision_recall(logits, mask)
                f1score = compute_f1score(precision, recall)
                accuracy = compute_accuracy(logits, mask)
                val_loss(loss)
                val_f1score(f1score)
                val_accuracy(accuracy)
                metrics = [('val_loss', loss), ("val_f1", f1score), ("val_acc", accuracy)]
                val_progbar.update(b + 1, metrics)

            for b, (image, mask) in enumerate(val_dataset):
                if b >= 3:
                    break
                logits = self.forward(image, training=False)
                save_validation(image[0], mask[0], logits[0], save_path, '_epoch_' + str(epoca + 1) + '_' + str(b))

            self.history['train_loss'].append(train_loss.result().numpy())
            self.history['train_acc'].append(train_accuracy.result().numpy())
            self.history['train_f1score'].append(train_f1score.result().numpy())

            self.history['val_loss'].append(val_loss.result().numpy())
            self.history['val_acc'].append(val_accuracy.result().numpy())
            self.history['val_f1score'].append(val_f1score.result().numpy())

            if self.history['val_f1score'][-1] >= best_f1score:
                self.save_variables()
                print("Model saved. f1score : {} --> {}".format(best_f1score, self.history['val_f1score'][-1]))
                best_f1score = self.history['val_f1score'][-1]

        plot(plot_path, self.name, self.history, epochs)

    def test(self, test_dataset, steps):
        if tf.executing_eagerly() is False:
            raise RuntimeError("evaluate method must be run only with eager execution.")
        self.load_variables()
        progbar = tf.keras.utils.Progbar(steps)
        save_path = create_folder_and_save_path('file/test/', self.name, split=False)
        count = 0
        for b, (image) in enumerate(test_dataset):
            logits = self.forward(image, training=False)
            progbar.update(b + 1)
            for logit in logits:
                lgt = create_label_mask(logit)
                prediction = (tf.keras.preprocessing.image.array_to_img(lgt))
                prediction.save(os.path.join(save_path,"test_"+str(count)+'.png'))
                count = count + 1

    def evaluate(self, test_dataset, steps):
        if tf.executing_eagerly() is False:
            raise RuntimeError("evaluate method must be run only with eager execution.")
        self.load_variables()
        precision_mean = tf.metrics.Mean('precision')
        recall_mean = tf.metrics.Mean('recall')
        accuracy_mean = tf.metrics.Mean('accuracy')
        f1score_mean = tf.metrics.Mean('f1score')
        progbar = tf.keras.utils.Progbar(steps)
        save_path = create_folder_and_save_path('file/predictions/', self.name, split=False)
        count = 0
        for b, (image, mask) in enumerate(test_dataset):
            labels = tf.cast(tf.squeeze(mask, axis=-1), tf.float32)
            logits = self.forward(image, training=False)

            precision, recall = precision_recall(logits, labels)
            accuracy = compute_accuracy(logits, labels)
            f1score = compute_f1score(precision, recall)
            count += 1
            if count < 50:
                for i in range(0, int(len(image) / 10)):
                    save_validation(image[i], mask[i], logits[i], save_path, count)
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











