from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
import keras.backend as K

import tensorflow as tf

# TODO TRY F1 AS METRIC IN MODEL.COMPILE
# TODO TRY OTHER ACTIVATION -> extract in arg of function
# TODO CHANGE LR OF ADAM
# PATCH SIZE USED WAS 96

#From old keras code
def get_f1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def a_unet(input_size, verbose = False):
    inputs = Input(input_size)
    s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
    
    dropout = 0.8 # TODO CHANGE
    alpha = 0.1 # TODO CHANGE
    
    # Encode
    
    conv1 = Conv2D(32, 3, activation=LeakyReLU(alpha), padding='same', kernel_initializer='he_normal')(s)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv2 = Conv2D(32, 3, activation=LeakyReLU(alpha), padding='same', kernel_initializer='he_normal')(conv1)
    conv2 = tf.keras.layers.BatchNormalization()(conv2)
    
    pool1 = tf.keras.layers.MaxPool2D(2)(conv2)
    pool1 = tf.keras.layers.Dropout(dropout)(pool1)
    
    
    conv3 = Conv2D(64, 3, activation=LeakyReLU(alpha), padding='same', kernel_initializer='he_normal')(pool1)
    conv3 = tf.keras.layers.BatchNormalization()(conv3)
    conv4 = Conv2D(64, 3, activation=LeakyReLU(alpha), padding='same', kernel_initializer='he_normal')(conv3)
    conv4 = tf.keras.layers.BatchNormalization()(conv4)
    
    pool2 = tf.keras.layers.MaxPool2D(2)(conv4)
    pool2 = tf.keras.layers.Dropout(dropout)(pool2)
    
    
    conv5 = Conv2D(128, 3, activation=LeakyReLU(alpha), padding='same', kernel_initializer='he_normal')(pool2)
    conv5 = tf.keras.layers.BatchNormalization()(conv5)
    conv6 = Conv2D(128, 3, activation=LeakyReLU(alpha), padding='same', kernel_initializer='he_normal')(conv5)
    conv6 = tf.keras.layers.BatchNormalization()(conv6)
    
    pool3 = tf.keras.layers.MaxPool2D(2)(conv6)
    pool3 = tf.keras.layers.Dropout(dropout)(pool3)
    
    
    conv7 = Conv2D(256, 3, activation=LeakyReLU(alpha), padding='same', kernel_initializer='he_normal')(pool3)
    conv7 = tf.keras.layers.BatchNormalization()(conv7)
    conv8 = Conv2D(256, 3, activation=LeakyReLU(alpha), padding='same', kernel_initializer='he_normal')(conv7)
    conv8 = tf.keras.layers.BatchNormalization()(conv8)
    
    pool4 = tf.keras.layers.MaxPool2D(2)(conv8)
    pool4 = tf.keras.layers.Dropout(dropout)(pool4)
    
    
    conv9 = Conv2D(512, 3, activation=LeakyReLU(alpha), padding='same', kernel_initializer='he_normal')(pool4)
    conv9 = tf.keras.layers.BatchNormalization()(conv9)
    conv10 = Conv2D(512, 3, activation=LeakyReLU(alpha), padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = tf.keras.layers.BatchNormalization()(conv10)
    
    pool5 = tf.keras.layers.MaxPool2D(2)(conv10)
    
    # Decode
    
    up1 = tf.keras.layers.Conv2DTranspose(256, 3, 2, padding='same', kernel_initializer='he_normal')(conv10)
    
    up1 = tf.keras.layers.concatenate([up1, conv8], axis=3)
    
    
    up2 = Conv2D(128, 3, activation=LeakyReLU(alpha), padding='same', kernel_initializer='he_normal')(up1)
    up2 = tf.keras.layers.BatchNormalization()(up2)
    up2 = Conv2D(128, 3, activation=LeakyReLU(alpha), padding='same', kernel_initializer='he_normal')(up2)
    up2 = tf.keras.layers.BatchNormalization()(up2)
    up2 = tf.keras.layers.Conv2DTranspose(128, 3, 2, padding='same', kernel_initializer='he_normal')(up2)
    up2 = tf.keras.layers.Dropout(dropout)(up2)
    
    up2 = tf.keras.layers.concatenate([up2, conv6], axis=3)
    
    
    up3 = Conv2D(64, 3, activation=LeakyReLU(alpha), padding='same', kernel_initializer='he_normal')(up2)
    up3 = tf.keras.layers.BatchNormalization()(up3)
    up3 = Conv2D(64, 3, activation=LeakyReLU(alpha), padding='same', kernel_initializer='he_normal')(up3)
    up3 = tf.keras.layers.BatchNormalization()(up3)
    up3 = tf.keras.layers.Conv2DTranspose(64, 3, 2, padding='same', kernel_initializer='he_normal')(up3)
    up3 = tf.keras.layers.Dropout(dropout)(up3)
    
    up3 = tf.keras.layers.concatenate([up3, conv4], axis=3)
    
    
    up4 = Conv2D(32, 3, activation=LeakyReLU(alpha), padding='same', kernel_initializer='he_normal')(up3)
    up4 = tf.keras.layers.BatchNormalization()(up4)
    up4 = Conv2D(32, 3, activation=LeakyReLU(alpha), padding='same', kernel_initializer='he_normal')(up4)
    up4 = tf.keras.layers.BatchNormalization()(up4)
    up4 = tf.keras.layers.Conv2DTranspose(32, 3, 2, padding='same', kernel_initializer='he_normal')(up4)
    up4 = tf.keras.layers.Dropout(dropout)(up4)
    
    up4 = tf.keras.layers.concatenate([up4, conv2], axis=3)
    
    
    up5 = Conv2D(32, 3, activation=LeakyReLU(alpha), padding='same', kernel_initializer='he_normal')(up4)
    up5 = tf.keras.layers.BatchNormalization()(up5)
    up5 = Conv2D(32, 3, activation=LeakyReLU(alpha), padding='same', kernel_initializer='he_normal')(up5)
    up5 = tf.keras.layers.BatchNormalization()(up5)

    outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(up5)
    
    model = Model(inputs = inputs, outputs = outputs, name='a_unet')

    model.compile(optimizer = Adam(learning_rate = 1e-4), loss = 'binary_crossentropy', metrics = [get_f1]) #TODO CHANGE
    
    if verbose:
        model.summary()

    return model
    

def fat_unet(input_size, verbose = False):
    #Taken from https://github.com/zhixuhao/unet/blob/master/model.py
    inputs = Input(input_size)
    s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
    
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(s)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10, name='fat_unet')

    model.compile(optimizer = Adam(learning_rate = 1e-4), loss = 'binary_crossentropy', metrics = [get_f1])
    
    if verbose:
        model.summary()

    return model

# PATCH SIZE USED WAS 80
def first_unet(input_size, verbose = False):
    # Taken from #https://github.com/bnsreenu/python_for_microscopists/blob/master/076-077-078-Unet_nuclei_tutorial.py
    inputs = tf.keras.layers.Input(input_size)
    s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = tf.keras.Model(inputs = inputs, outputs = outputs, name = 'first_unet')
    
    model.compile(optimizer = "adam", loss='binary_crossentropy', metrics=['accuracy'])
    
    if verbose:
        model.summary()
        
    return model

def short_unet(input_size, verbose=False):   
    #Taken from https://nbviewer.org/github/ashishpatel26/Semantic-Segmentation-Keras-Tensorflow-Example/blob/main/Areal_Image_segmentation_with_a_U_Net_like_architecture.ipynb
    
    inputs = tf.keras.layers.Input(input_size)

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = tf.keras.layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = tf.keras.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.UpSampling2D(2)(x)

        # Project residual
        residual = tf.keras.layers.UpSampling2D(2)(previous_block_activation)
        residual = tf.keras.layers.Conv2D(filters, 1, padding="same")(residual)
        x = tf.keras.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid", padding="same")(x)

    # Define the model
    model = tf.keras.Model(inputs, outputs)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=[get_f1])
    
    if verbose:
        model.summary()
        
    return model


def multi_unet_model(input_size, verbose = False):
    #Build the model
    inputs = tf.keras.layers.Input(input_size)
    s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

    #Contraction path
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    #Expansive path 
    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name="multi_unet_model")
    
    model.compile(optimizer = "adam", loss='binary_crossentropy', metrics=['accuracy'])
    
    if verbose:
        model.summary()
        
    return model

def variation_unet(input_size, verbose = False):
    #Taken from https://medium.com/@nithishmailme/satellite-imagery-road-segmentation-ad2964dc3812
    #Build the model
    inputs = tf.keras.layers.Input(input_size)
    s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

    x = Conv2D(32, (3, 3), padding='same', name='block1_conv1')(s)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), padding='same', name='block1_conv2')(x)
    x = BatchNormalization()(x)
    block_1_out = Activation('relu')(x)
    x = MaxPooling2D()(block_1_out)
    x = Dropout(0.2)(x)

    # Block 2
    x = Conv2D(64, (3, 3), padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same', name='block2_conv2')(x)
    x = BatchNormalization()(x)
    block_2_out = Activation('relu')(x)
    x = MaxPooling2D()(block_2_out)
    x = Dropout(0.2)(x)

    # Block 3
    x = Conv2D(128, (3, 3), padding='same', name='block3_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same', name='block3_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same', name='block3_conv3')(x)
    x = BatchNormalization()(x)
    block_3_out = Activation('relu')(x)
    x = MaxPooling2D()(block_3_out)
    x = Dropout(0.2)(x)

    # Block 4
    x = Conv2D(256, (3, 3), padding='same', name='block4_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same', name='block4_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same', name='block4_conv3')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    block_4_out = Activation('relu')(x)

    x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', name = 'Conv2DTranspose_UP2')(block_4_out)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Concatenate()([x, block_3_out])
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)

    # UP 3
    x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', name = 'Conv2DTranspose_UP3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Concatenate()([x, block_2_out])
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)

    # UP 4
    x = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', name = 'Conv2DTranspose_UP4')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Concatenate()([x, block_1_out])
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    outputs = Conv2D(2, (3, 3), activation='softmax', padding='same')(x)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name="variation_unet")
    
    model.compile(optimizer = "adam", loss='binary_crossentropy', metrics=['accuracy'])
    
    if verbose:
        model.summary()
        
    return model