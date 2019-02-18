import tensorflow as tf
from tensorflow.keras.layers import (Activation, AveragePooling2D,
                                            BatchNormalization, Conv2D, Conv3D,
                                            Dense, Flatten,
                                            GlobalAveragePooling2D,
                                            GlobalMaxPooling2D, Input,
                                            MaxPooling2D, MaxPooling3D,
                                            Reshape, Dropout, concatenate,
											UpSampling2D)
from tensorflow.keras import applications, regularizers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K_B

class CSRNet(object):
    """Fully convolutional autoencoder for temporal-regularity detection.
        For simplicity, fully convolutional autoencoder structure is
        changed to be fixed as symmetric.

        Reference:
        [1] Learning Temporal Regularity in Video Sequences
            (http://arxiv.org/abs/1604.04574)
        [2] https://github.com/carpedm20/DCGAN-tensorflow/blob/master/ops.py
    """

    def _preprocess_input(self, x):
        """Preprocesses a tensor encoding a batch of images.

        # Arguments
            x: input Numpy tensor, 4D.
            data_format: data format of the image tensor.

        # Returns
            Preprocessed tensor.
        """

        # 'RGB'->'BGR'
        x = x[:, :, ::-1]
        # Zero-center by mean pixel
        x = x - tf.stack((tf.ones_like(x[:, :, :, 0]) * tf.constant(103.939),
                          tf.ones_like(x[:, :, :, 1]) * tf.constant(116.779)
                          , tf.ones_like(x[:, :, :, 2]) * tf.constant(123.68)), axis=-1)

        # x = 2*x/255
        return x


    def __init__(self, cfg, images, batch_size, type='a'):
        """
        Args:
          sess : TensorFlow session
          input_shape : Shape of the input data. [batch_size, clip_length, h, w, 1]
        """

        self._x = self._preprocess_input(images)
        self._batch_size = batch_size
        # self._h = cfg.height
        # self._w = cfg.width
        # self._c = self._x.get_shape().as_list()[-1]
        with tf.variable_scope('front'):
           front = self._build_front(self._x)
        with tf.variable_scope('backend'):
            self.full_model = self._build_backend(front, type=type)
        self.output = self.full_model.output


    def _build_front(self, input_images):
        base_model = applications.VGG16(input_tensor=input_images, weights='imagenet',
                                        include_top=False, input_shape=(256, 256, 3))
        BOTTLENECK_TENSOR_NAME = 'block4_conv3'  # This is the 13th layer in VGG16
        front = self._create_non_trainable_model(base_model, BOTTLENECK_TENSOR_NAME)  # Frontend
        return front


    def _create_non_trainable_model(self, base_model, BOTTLENECK_TENSOR_NAME,
                                   use_global_average=False):
        '''
        Parameters
        ----------
        base_model: This is the pre-trained base model with which the non-trainable model is built
        Note: The term non-trainable can be confusing. The non-trainable-parametes are present only in this
        model. The other model (trianable model doesnt have any non-trainable parameters). But if you chose to
        omit the bottlenecks due to any reason, you will be training this network only. (If you choose
        --omit_bottleneck flag). So please adjust the place in this function where I have intentionally made
        certain layers non-trainable.
        Returns
        -------
        non_trainable_model: This is the model object which is the modified version of the base_model that has
        been invoked in the beginning. This can have trainable or non trainable parameters. If bottlenecks are
        created, then this network is completely non trainable, (i.e) this network's output is the bottleneck
        and the network created in the trainable is used for training with bottlenecks as input. If bottlenecks
        arent created, then this network is trained. So please use accordingly.
        '''
        # This post-processing of the deep neural network is to avoid memory errors
        x = (base_model.get_layer(BOTTLENECK_TENSOR_NAME))
        all_layers = base_model.layers
        for i in range(base_model.layers.index(x)):
            all_layers[i].trainable = False
        mid_out = base_model.layers[base_model.layers.index(x)]
        non_trainable_model = Model(base_model.input, mid_out.output)
        # non_trainable_model = Model(inputs = base_model.input, outputs = [x])

        # for layer in non_trainable_model.layers:
        #     layer.trainable = False
        return (non_trainable_model)


    def _build_backend(self, front, type='a'):
        if type == 'a':
            full_model = self._backend_A(front)
        elif type == 'b':
            full_model = self._backend_B(front)
        elif type == 'c':
            full_model = self._backend_C(front)
        elif type == 'd':
            full_model = self._backend_D(front)
        else:
            print('[!!!] wrong type')
        return full_model


    def _backend_A(self, front, weights=None):
        x = Conv2D(512, 3, padding='same', dilation_rate=1,
                   kernel_regularizer=regularizers.l2(0.01), name="dil_A1")(front.output)
        x = Activation('relu')(x)
        x = Conv2D(512, 3, padding='same', dilation_rate=1,
                   kernel_regularizer=regularizers.l2(0.01), name="dil_A2")(x)
        x = Activation('relu')(x)
        x = Conv2D(512, 3, padding='same', dilation_rate=1,
                   kernel_regularizer=regularizers.l2(0.01), name="dil_A3")(x)
        x = Activation('relu')(x)
        x = Conv2D(256, 3, padding='same', dilation_rate=1,
                   kernel_regularizer=regularizers.l2(0.01), name="dil_A4")(x)
        x = Activation('relu')(x)
        x = Conv2D(128, 3, padding='same', dilation_rate=1,
                   kernel_regularizer=regularizers.l2(0.01), name="dil_A5")(x)
        x = Activation('relu')(x)
        x = Conv2D(64, 3, padding='same', dilation_rate=1, kernel_regularizer=regularizers.l2(0.01),
                   name="dil_A6")(x)
        x = Activation('relu')(x)

        x = Conv2D(1, 1, padding='same', dilation_rate=1, name="dil_A7")(x)

        model = Model(front.input, x, name="Transfer_learning_model")
        return (model)

    def _backend_B(self, f, weights=None):

        x = Conv2D(512, 3, padding='same', dilation_rate=2, activation='relu', name="dil_B1")(
            f.output)
        # x = BatchNormalization(name='bn_b1')(x)
        x = Conv2D(512, 3, padding='same', dilation_rate=2, activation='relu', name="dil_B2")(x)
        # x = BatchNormalization(name='bn_b2')(x)
        x = Conv2D(512, 3, padding='same', dilation_rate=2, activation='relu', name="dil_B3")(x)
        # x = BatchNormalization(name='bn_b3')(x)
        x = Conv2D(256, 3, padding='same', dilation_rate=2, activation='relu', name="dil_B4")(x)
        # x = BatchNormalization(name='bn_b4')(x)
        x = Conv2D(128, 3, padding='same', dilation_rate=2, activation='relu', name="dil_B5")(x)
        # x = BatchNormalization(name='bn_b5')(x)
        x = Conv2D(64, 3, padding='same', dilation_rate=2, activation='relu', name="dil_B6")(x)

        x = Conv2D(1, 1, padding='same', dilation_rate=1, name="dil_B7")(x)
        model = Model(f.input, x, name="Transfer_learning_model")
        return (model)

    def _backend_C(self, f, weights=None):

        x = Conv2D(512, 3, padding='same', dilation_rate=2, activation='relu', name="dil_C1")(
            f.output)
        x = Conv2D(512, 3, padding='same', dilation_rate=2, activation='relu', name="dil_C2")(x)
        x = Conv2D(512, 3, padding='same', dilation_rate=2, activation='relu', name="dil_C3")(x)
        x = Conv2D(256, 3, padding='same', dilation_rate=4, activation='relu', name="dil_C4")(x)
        x = Conv2D(128, 3, padding='same', dilation_rate=4, activation='relu', name="dil_C5")(x)
        x = Conv2D(64, 3, padding='same', dilation_rate=4, activation='relu', name="dil_C6")(x)

        x = Conv2D(1, 1, padding='same', dilation_rate=1, name="dil_C7")(x)
        model = Model(f.input, x, name="Transfer_learning_model")
        return (model)

    def _backend_D(self, f, weights=None):

        x = Conv2D(512, 3, padding='same', dilation_rate=4, activation='relu', name="dil_D1")(
            f.output)
        x = Conv2D(512, 3, padding='same', dilation_rate=4, activation='relu', name="dil_D2")(x)
        x = Conv2D(512, 3, padding='same', dilation_rate=4, activation='relu', name="dil_D3")(x)
        x = Conv2D(256, 3, padding='same', dilation_rate=4, activation='relu', name="dil_D4")(x)
        x = Conv2D(128, 3, padding='same', dilation_rate=4, activation='relu', name="dil_D5")(x)
        x = Conv2D(64, 3, padding='same', dilation_rate=4, activation='relu', name="dil_D6")(x)

        x = Conv2D(1, 1, padding='same', dilation_rate=1, name="dil_D7")(x)
        model = Model(f.input, x, name="Transfer_learning_model")
        return (model)


    def _lrelu(self, x, leak=0.2, name="lrelu"):
        return tf.maximum(x, leak * x)
