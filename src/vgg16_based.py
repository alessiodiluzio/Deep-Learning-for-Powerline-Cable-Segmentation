import tensorflow as tf


class Conv2DLayer(tf.keras.Model):

    def __init__(self, filters, kernel_size, strides, padding, activation, kernel_initializer, dilation_rate=1):
        super(Conv2DLayer, self).__init__(name='conv_block')
        self.convolution = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                                                  padding=padding, activation=activation,
                                                  kernel_initializer=kernel_initializer,dilation_rate=dilation_rate)
        self.batch_normalization = tf.keras.layers.BatchNormalization(axis=-1)

    def __call__(self, input_tensor, training=False):
        x = self.convolution(input_tensor)
        x = self.batch_normalization(x, training=training)
        return tf.nn.relu(x)


class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__(name='Encoder')

        base_model = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=None,
                                                       input_shape=(128, 128, 3), pooling=None, classes=2)
        base_model.trainable = False
        layer_names = [
            'block1_conv1',  # 64x64
            'block1_conv2',  # 32x32
            'block1_pool',  # 16x16
            'block2_conv1',  # 8x8
            'block2_conv2',  # 4x4
        ]
        layers = [base_model.get_layer(name) for name in layer_names]
        for layer in layers:
            layer.output.trainable = False
        self.conv_1 = layers[0]
        self.conv_2 = layers[1]
        self.max_pool = layers[2]
        self.conv_3 = layers[3]
        self.conv_4 = layers[4]
        self.skip_connections = {}

    def __call__(self, input_tensor, training):

        self.skip_connections = {0: []}

        # First Downsampling block

        x = self.conv_1(input_tensor, training=training)
        x = self.conv_2(x, training=training)
        self.skip_connections[0].append(x)
        x = self.max_pool(x)

        # Second Downsampling block

        x = self.conv_3(x, training=training)
        x = self.conv_4(x, training=training)

        return x


class SkipConnection(tf.keras.Model):

    def __init__(self):
        super(SkipConnection, self).__init__(name='skip_conn')
        self.up = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.concatenate = tf.keras.layers.Concatenate(axis=-1)

    def __call__(self, input_tensor_1, input_tensor_2):
        if not isinstance(input_tensor_2, list):
            input_tensor_2 = [input_tensor_2]
        input_tensor_1 = [self.up(input_tensor_1)]
        maps = input_tensor_1 + input_tensor_2
        x = self.concatenate(maps)
        return x


class Decoder(tf.keras.Model):

    def __init__(self, skips=True):
        super(Decoder, self).__init__(name='Decoder')

        self.dil_conv_1 = Conv2DLayer(filters=128, kernel_size=3, strides=1, padding="same", activation='relu',
                                      kernel_initializer='he_normal')
        self.dil_conv_2 = Conv2DLayer(filters=128, kernel_size=3, strides=1, padding="same", activation='relu',
                                      kernel_initializer='he_normal',dilation_rate=2)
        self.dil_conv_3 = Conv2DLayer(filters=128, kernel_size=3, strides=1, padding="same", activation='relu',
                                      kernel_initializer='he_normal',dilation_rate=4)
        self.dil_conv_4 = Conv2DLayer(filters=64, kernel_size=3, strides=1, padding="same", activation='relu',
                                      kernel_initializer='he_normal')
        self.skips = skips

        if self.skips:
            self.up_1 = SkipConnection()
        else:
            self.up_1 = tf.keras.layers.UpSampling2D(size=(2, 2))

        self.conv_4 = Conv2DLayer(filters = 64, kernel_size = 3, strides = 1, padding ="same"
                                  , activation="relu", kernel_initializer='he_normal')


        self.final_conv = tf.keras.layers.Conv2D(filters=2, kernel_size=1, strides=1, padding='same', activation=None)
        # Conv2DLayer(filters=2,  kernel_size=1, strides=1, padding='same',
        # activation=None, kernel_initializer="he_normal")

    def __call__(self, input_tensor, training, skip_connections):

        # First Upsampling block

        x = self.dil_conv_1(input_tensor, training=training)
        x = self.dil_conv_2(x, training=training)
        x = self.dil_conv_3(x, training=training)
        x = self.dil_conv_4(x, training=training)
        if self.skips:
            x = self.up_1(x, skip_connections[0])
        else:
            x = self.up_1(x)
        x = self.conv_4(x, training=training)
        x = self.final_conv(x)

        return x


class DeepDecoder(tf.keras.Model):

    def __init__(self, skips=True):
        super(DeepDecoder, self).__init__(name='Decoder')

        self.dil_conv_1 = Conv2DLayer(filters=64, kernel_size=3, strides=1, padding="same", activation='relu',
                                      kernel_initializer='he_normal')
        self.dil_conv_2 = Conv2DLayer(filters=128, kernel_size=3, strides=1, padding="same", activation='relu',
                                      kernel_initializer='he_normal', dilation_rate=2)
        self.dil_conv_3 = Conv2DLayer(filters=256, kernel_size=3, strides=1, padding="same", activation='relu',
                                      kernel_initializer='he_normal', dilation_rate=4)
        self.dil_conv_4 = Conv2DLayer(filters=256, kernel_size=3, strides=1, padding="same", activation='relu',
                                      kernel_initializer='he_normal', dilation_rate=8)
        self.dil_conv_5 = Conv2DLayer(filters=128, kernel_size=3, strides=1, padding="same", activation='relu',
                                      kernel_initializer='he_normal', dilation_rate=16)
        self.dil_conv_6 = Conv2DLayer(filters=64, kernel_size=3, strides=1, padding="same", activation='relu',
                                      kernel_initializer='he_normal')
        if skips:
            self.up_1 = SkipConnection()
        else:
            self.up_1 = tf.keras.layers.UpSampling2D(size=(2, 2))

        self.conv_4 = Conv2DLayer(filters=64, kernel_size=3, strides=1, padding="same"
                                  , activation="relu", kernel_initializer='he_normal')

        self.final_conv = tf.keras.layers.Conv2D(filters=2, kernel_size=1, strides=1, padding='same', activation=None,
                                                 kernel_initializer="he_normal")

    def __call__(self, input_tensor, training, skip_connections):

        # First Upsampling block

        x = self.dil_conv_1(input_tensor, training=training)
        x = self.dil_conv_2(x, training=training)
        x = self.dil_conv_3(x, training=training)
        x = self.dil_conv_4(x, training=training)
        x = self.dil_conv_5(x, training=training)
        x = self.dil_conv_6(x, training=training)
        x = self.up_1(x, skip_connections[0])
        x = self.conv_4(x, training=training)
        x = self.final_conv(x)

        return x

