import keras
import logging
import numpy
from core_multi.utils import Utils
from core_multi.layers import DeprocessStylizedImage
from core_multi.reflection_padding import ReflectionPadding2D
from core_multi.conditional_instance_norm import ConditionalInstanceNormalization
from keras import backend as K

logger = logging.getLogger(__name__)


class MultiStyleTransferNetwork(object):
    """A class that builds a Keras model to perform style transfer.

    The architecture for this model comes from Johnson et al:
    https://arxiv.org/abs/1603.08155
    https://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf

    It differs slightly from Johnson's model by swapping reflective
    padding with Zero Padding and Batch Normalization for
    Instance Normalization as recommended in Ulyanov et al:
    https://arxiv.org/abs/1607.08022
    """
    @classmethod
    def setSW(cls, sw):
        K.set_value(cls.sw, sw)

    @classmethod
    def setStyleIndex(cls, style_index):
        style_shape = K.int_shape(cls.sw)
        if style_index <= (style_shape[-1] - 1):
            sw = numpy.zeros(style_shape, dtype=numpy.float32)
            sw[0, style_index] = 1
            cls.setSW(sw)
        else:
            logger.error("Style index overflow: ", style_index)

    @classmethod
    def build(
            cls,
            image_size,
            style_num,
            alpha=1.0,
            input_tensor=None,
            checkpoint_file=None,
            activation='sigmoid'):
        """Build a Transfer Network Model using keras' functional API.

        Args:
            image_size - the size of the input and output image (H, W)
            alpha - a width parameter to scale the number of channels by

        Returns:
            model: a keras model object
        """
        if activation != 'sigmoid' and activation != 'tanh' :
            logger.error("Only sigmoid or tanh activation supported")
            exit()
        else:
            logger.info("Using %s activation for output image" % activation)

        if style_num < 1:
            logger.error("Style number should >= 1")
            exit()

        # Init and select first style
        style_shape = (1, style_num)
        style_weights = numpy.zeros(style_shape, dtype=numpy.float32)
        style_weights[0, 0] = 1
        cls.style_num = style_num
        cls.sw = K.variable(style_weights)

        x = keras.layers.Input(
            shape=(image_size[0], image_size[1], 3), tensor=input_tensor)
        out = cls._convolution(x, int(alpha * 32), 9, cls.sw, strides=1)
        out = cls._convolution(out, int(alpha * 64), 3, cls.sw, strides=2)
        out = cls._convolution(out, int(alpha * 128), 3, cls.sw, strides=2)
        out = cls._residual_block(out, int(alpha * 128), cls.sw)
        out = cls._residual_block(out, int(alpha * 128), cls.sw)
        out = cls._residual_block(out, int(alpha * 128), cls.sw)
        out = cls._residual_block(out, int(alpha * 128), cls.sw)
        out = cls._residual_block(out, int(alpha * 128), cls.sw)
        out = cls._upsample(out, int(alpha * 64), 3, cls.sw)
        out = cls._upsample(out, int(alpha * 32), 3, cls.sw)
        # Add a layer of padding to keep sizes consistent.
        out = cls._convolution(out, 3, 9, cls.sw, relu=False, padding='valid')
        # Restrict outputs of pixel values to -1 and 1.
        out = keras.layers.Activation(activation)(out)
        # Deprocess the image into valid image data. Note we'll need to define
        # a custom layer for this in Core ML as well.
        out = DeprocessStylizedImage(activation=activation)(out)
        model = keras.models.Model(inputs=x, outputs=out)

        # Optionally load weights from a checkpoint
        if checkpoint_file:
            logger.info(
                'Loading weights from checkpoint: %s' % checkpoint_file
            )
            if checkpoint_file.startswith('gs://'):
                checkpoint_file = Utils.copy_file_from_gcs(checkpoint_file)
            # Change by_name to False because the names of layers may differ
            model.load_weights(checkpoint_file, by_name=False)
        return model

    @classmethod
    def _conv_only(
            cls, x, n_filters, kernel_size, strides=1,
            padding='valid', use_bias=False):
        """Create a convolution block.

        This block consists of a convolution layer, normalization, and an
        optional RELU activation.

        Args:
            x - a keras layer as input
            n_filters - the number of output dimensions
            kernel_size - an integer or tuple specifying the (width, height) of
                         the 2D convolution window
            strides - An integer or tuple/list of 2 integers, specifying the
                      strides of the convolution along the width and height.
                      Default 1.
            padding: one of "valid" or "same" (case-insensitive).
            relu - a bool specifying whether or not a RELU activation is
                   applied. Default True.
            use_bias = a bool specifying whether or not to use a bias term
        """
        padding_size = int((kernel_size - 1)/2)
        padded_inputs= ReflectionPadding2D(padding=(padding_size,padding_size))(x)
        out = keras.layers.convolutional.Conv2D(
            n_filters,
            kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias
        )(padded_inputs)

        return out

    @classmethod
    def _convolution(
            cls, x, n_filters, kernel_size, sw, strides=1,
            padding='valid', relu=True, use_bias=False):
        """Create a convolution block.

        This block consists of a convolution layer, normalization, and an
        optional RELU activation.

        Args:
            x - a keras layer as input
            n_filters - the number of output dimensions
            kernel_size - an integer or tuple specifying the (width, height) of
                         the 2D convolution window
            strides - An integer or tuple/list of 2 integers, specifying the
                      strides of the convolution along the width and height.
                      Default 1.
            padding: one of "valid" or "same" (case-insensitive).
            relu - a bool specifying whether or not a RELU activation is
                   applied. Default True.
            use_bias = a bool specifying whether or not to use a bias term
        """
        padding_size = int((kernel_size - 1) / 2)
        padded_inputs = ReflectionPadding2D(padding=(padding_size, padding_size))(x)
        out = keras.layers.convolutional.Conv2D(
            n_filters,
            kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias
        )(padded_inputs)

        # We are using the keras-contrib library from @farizrahman4u for
        # an implementation of Instance Normalization. Note here that we are
        # specifying the normalization axis to be -1, or the channel axis.
        # By default this is None and simple Batch Normalization is applied.
        out = ConditionalInstanceNormalization(axis=-1, epsilon=1e-10, sw=sw)(out)
        if relu:
            out = keras.layers.Activation('relu')(out)
        return out

    @classmethod
    def _residual_block(cls, x, n_filters, sw, kernel_size=3):
        """Construct a residual block.

        Args:
            x - a keras layer as input
            n_filters - the number of output dimensions
            kernel_size - an integer or tuple specifying the (width, height) of
                         the 2D convolution window. Default 3.
        Returns:
            out - a keras layer as output
        """
        out = cls._convolution(x, n_filters, kernel_size, sw, padding='valid')
        out = cls._convolution(
            out, n_filters, kernel_size, sw, padding='valid', relu=False
        )

        out = keras.layers.Add()([out, x])
        return out

    @classmethod
    def _upsample(cls, x, n_filters, kernel_size, sw, size=2):
        """Construct an upsample block.

        Args:
            x - a keras layer as input
            n_filters - the number of output dimensions
            kernel_size - an integer or tuple specifying the (width, height) of
                         the 2D convolution window. Default 3.
        Returns:
            out - a keras layer as output
        """
        out = keras.layers.UpSampling2D(size=size)(x)
        out = cls._convolution(out, n_filters, kernel_size, sw, padding='valid')
        return out


class IntermediateVGG(object):
    """A VGG network class that allows easy access to intermediate layers.

    This class takes the default VGG16 application packaged with Keras and
    constructs a dictionary mapping layer names to layout puts so that
    we can easily extract the network's features at any level. These outputs
    are used to compute losses in artistic style transfer.

    """

    def __init__(self, prev_layer=None, input_tensor=None):
        """Initialize the model.

        Args:
            prev_layer - a keras layer to use as an input layer to the
                         VGG model. This allows us to stitch other models
                         together with the VGG.
            input_tensor - a tensor that will be used as input for the
                          VGG.
        """
        # Create the Keras VGG Model
        self.model = keras.applications.vgg16.VGG16(
            weights='imagenet',
            include_top=False,
            input_tensor=input_tensor
        )

        # Make sure none of the VGG layers are trainable
        for layer in self.model.layers:
            layer.trainable = False

        # if a previous layer is specified, stitch that layer to the
        # input of the VGG model and rewire the entire model.
        self.layers = {}
        if prev_layer is not None:
            # We need to apply all layers to the output of the style net
            in_layer = prev_layer
            for layer in self.model.layers[1:]:  # Ignore the input layer
                in_layer = layer(in_layer)
                self.layers[layer.name] = in_layer
        else:
            self.layers = dict(
                [(layer.name, layer.output) for layer in self.model.layers]
            )
