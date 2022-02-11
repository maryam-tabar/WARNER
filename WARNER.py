# This code is adopted from https://github.com/raghakot/keras-resnet

from __future__ import division
import six
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    Concatenate,
    Multiply,
    Add
)
from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers.merge import add
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from keras.regularizers import l2
from keras import backend as K
from pconv_layer import PConv2D #obtained from https://github.com/MathiasGruber/PConv-Keras/tree/master/libs



def county_conv_net3(**conv_params):
    kernel_size = conv_params["kernel_size"]
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(inputs):
        conv1_1, mask1_1 = PConv2D(filters=4, n_channels = 3, kernel_size=kernel_size,
                      strides=(4, 4), padding="valid",
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)([inputs[0], inputs[3]])
        conv1_1 = Activation("relu")(conv1_1)
        
        conv1_2, mask1_2 = PConv2D(filters=4, n_channels = 3, kernel_size=kernel_size,
                      strides=(4, 4), padding="valid",
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)([inputs[1], inputs[3]])
        conv1_2 = Activation("relu")(conv1_2)
        
        conv1_3, mask1_3 = PConv2D(filters=4, n_channels = 3, kernel_size=kernel_size,
                      strides=(4, 4), padding="valid",
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)([inputs[2], inputs[3]])
        conv1_3 = Activation("relu")(conv1_3)
        
        
        mask1 = Concatenate(axis=1)([mask1_1, mask1_2, mask1_3])
        conv1 = Concatenate(axis=1)([conv1_1, conv1_2, conv1_3])
        
        
        
        conv2, mask2 = PConv2D(filters=16, kernel_size=kernel_size,
                      strides=(4, 4), padding="valid",
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)([conv1, mask1])
                      
        conv2 = _bn_relu(conv2)
        
        
        conv3, mask3 = PConv2D(filters=32, kernel_size=kernel_size,
                      strides=(4, 4), padding="valid",
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)([conv2, mask2])
                      
        conv3 = _bn_relu(conv3)
        
        conv4, mask4 = PConv2D(filters=64, kernel_size=kernel_size,
                      strides=(4, 4), padding="valid",
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)([conv3, mask3])
                      
        conv4 = _bn_relu(conv4)
        

        conv_mult = Multiply()([conv4, mask4])
        block_shape = K.int_shape(conv_mult)
        pool_county = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
                                 strides=(1, 1), data_format = 'channels_first')(conv_mult)
        conv_flatten = Flatten()(pool_county)
        dense_county = Dense(units=64, activation="relu")(conv_flatten)
        
        return dense_county

    return f




def _bn_relu(input):
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)



def _conv_bn_relu(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(inputs):
        output = PConv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(inputs)
        return [_bn_relu(output[0]), output[1]]

    return f



def _bn_relu_conv(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(inputs):
        mask = inputs[1]
        activation = _bn_relu(inputs[0])
        output =  PConv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)([activation, mask])
                      
                      
       	return output

    return f
    


def _shortcut(input, mask, residual):
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]
    
    
    shortcut = [input, mask]
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = PConv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))([input, mask])

    return add([shortcut[0], residual])


def _residual_block(block_function, filters, repetitions, is_first_layer=False):
    def f(inputs):
        input_img, input_mask = inputs[0], inputs[1]
        
        for i in range(repetitions):
            init_strides = (1, 1)
            if i == 0 and not is_first_layer:
                init_strides = (2, 2)
            
            outputs = block_function(filters=filters, init_strides=init_strides,
              is_first_block_of_first_layer=(is_first_layer and i == 0))([input_img, input_mask])
                                   
            input_img, input_mask = outputs[0], outputs[1]
            
            
        return [input_img, input_mask]

    return f
    
    

def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    def f(inputs):
        input = inputs[0]
        mask = inputs[1]
        
        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_2d1 = PConv2D(filters=filters, kernel_size=(3, 3),
                           strides=init_strides,
                           padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=l2(1e-4))(inputs)
       
        else:
            conv_2d1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                  strides=init_strides)(inputs)

        residual_output = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv_2d1)
        return [_shortcut(input, mask, residual_output[0]), residual_output[1]]

    return f
    


def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    
    if K.image_data_format() == 'tf':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3


def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


class WARNER(object):
    @staticmethod
    def build(input_shape, input_county_shape, block_fn, repetitions):
        _handle_dim_ordering()
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        # Permute dimension order if necessary
        if K.image_data_format() == 'tf':
            input_shape = (input_shape[1], input_shape[2], input_shape[0])

        # Load function from str if needed.
        block_fn = _get_block(block_fn)

        input_county17 = Input(shape=input_county_shape) # satellite imagery of counties in 2017
        input_county18 = Input(shape=input_county_shape)# satellite imagery of counties in 2018
        input_county19 = Input(shape=input_county_shape)# satellite imagery of counties in 2019
        mask_county_i = Input(shape=input_county_shape)# mask matrices
        
        county_features = county_conv_net3(kernel_size=(7, 7))([input_county17, input_county18, input_county19, mask_county_i])
        
        
        input17 = Input(shape=input_shape) # satellite imagery of census tracts in 2017
        input18 = Input(shape=input_shape)# satellite imagery of census tracts in 2018
        input19 = Input(shape=input_shape)# satellite imagery of census tracts in 2019
        input = Concatenate(axis=1)([input17, input18, input19])
        mask_i = Input(shape=input_shape)# mask matrices
        mask = Concatenate(axis=1)([mask_i, mask_i, mask_i])
        conv1, mask1 = _conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))([input, mask])
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same", data_format = 'channels_first')(conv1)
        mask_pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same", data_format = 'channels_first')(mask1)

        block = pool1
        mask_block = mask_pool1
        filters = 64
        
        for i, r in enumerate(repetitions):
            block, mask_block = _residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))([block, mask_block])
            filters *= 2
            
        # Last activation
        block = _bn_relu(block)

        # Classifier block
        final_block = Multiply()([block, mask_block])
        block_shape = K.int_shape(final_block)
        pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
                                 strides=(1, 1), data_format = 'channels_first')(final_block)
                                 
        flatten_block = Flatten()(pool2)
        
        final_layer = Concatenate()([flatten_block,county_features])
        final_layer = Dense(units=128, activation="relu")(final_layer)
        dense = Dense(units=1, activation="sigmoid")(final_layer)

        model = Model(inputs=[input_county17, input_county18, input_county19, mask_county_i, input17,input18,input19, mask_i], outputs=dense)
        return model

    @staticmethod
    def build_warner(input_shape, input_county_shape):
        return WARNER.build(input_shape, input_county_shape, basic_block, [2, 2, 2])
