from __future__ import absolute_import
__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'

from keras.layers import Input, Dense, Dropout, \
                         RepeatVector, LSTM, concatenate, \
                         Conv2D, MaxPooling2D, Flatten, GRU
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras import *
from .Config import *
from .AModel import *
from keras.callbacks import ModelCheckpoint

from keras.applications.xception import Xception

class pix2code(AModel):
    def __init__(self, input_shape, output_size, output_path):
        AModel.__init__(self, input_shape, output_size, output_path)
        self.name = "pix2code"

        image_model = Sequential()

        xception = Xception(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
        xception.summary()

        first_six_xception_modules = xception.layers[1:66] # first 16 layer handle input, afterwards, 10 layers per module

        for layer in first_six_xception_modules:
            layer.trainable = False

        xception.layers.pop() # remove last layer

        image_model.add(xception)
        image_model.add(Dense(1024, activation='relu'))
        image_model.add(Dropout(0.3))

        # image_model.add(Conv2D(32, (3, 3), padding='valid', activation='relu', input_shape=input_shape))
        # image_model.add(Conv2D(32, (3, 3), padding='valid', activation='relu'))
        # image_model.add(MaxPooling2D(pool_size=(2, 2)))
        # image_model.add(Dropout(0.25))

        # image_model.add(Conv2D(64, (3, 3), padding='valid', activation='relu'))
        # image_model.add(Conv2D(64, (3, 3), padding='valid', activation='relu'))
        # image_model.add(MaxPooling2D(pool_size=(2, 2)))
        # image_model.add(Dropout(0.25))

        # image_model.add(Conv2D(128, (3, 3), padding='valid', activation='relu'))
        # image_model.add(Conv2D(128, (3, 3), padding='valid', activation='relu'))
        # image_model.add(MaxPooling2D(pool_size=(2, 2)))
        # image_model.add(Dropout(0.25))

        # image_model.add(Flatten())
        # image_model.add(Dense(1024, activation='relu'))
        # image_model.add(Dropout(0.3))
        # image_model.add(Dense(1024, activation='relu'))
        # image_model.add(Dropout(0.3))

        image_model.add(RepeatVector(CONTEXT_LENGTH))

        visual_input = Input(shape=input_shape)
        encoded_image = image_model(visual_input)

        language_model = Sequential()
        # language_model.add(LSTM(256, return_sequences=True, input_shape=(CONTEXT_LENGTH, output_size)))
        # language_model.add(LSTM(256, return_sequences=True))


        # language_model.add(LSTM(128, return_sequences=True, input_shape=(CONTEXT_LENGTH, output_size)))
        # language_model.add(LSTM(256, return_sequences=True))

        # language_model.add(LSTM(192, return_sequences=True, input_shape=(CONTEXT_LENGTH, output_size)))
        # language_model.add(LSTM(192, return_sequences=True))
    
        # language_model.add(GRU(256, return_sequences=True, input_shape=(CONTEXT_LENGTH, output_size)))
        # language_model.add(GRU(256, return_sequences=True))

        # language_model.add(LSTM(128, return_sequences=True, input_shape=(CONTEXT_LENGTH, output_size)))
        # language_model.add(LSTM(128, return_sequences=True))

        language_model.add(GRU(128, return_sequences=True, input_shape=(CONTEXT_LENGTH, output_size)))
        language_model.add(GRU(128, return_sequences=True))

        textual_input = Input(shape=(CONTEXT_LENGTH, output_size))
        encoded_text = language_model(textual_input)

        decoder = concatenate([encoded_image, encoded_text])

        # decoder = LSTM(512, return_sequences=True)(decoder)
        # decoder = LSTM(512, return_sequences=False)(decoder)


        #decoder = GRU(386, return_sequences=True)(decoder)
        #decoder = GRU(386, return_sequences=False)(decoder)

        # decoder = LSTM(512, return_sequences=True)(decoder)
        # decoder = LSTM(1024, return_sequences=False)(decoder)
        
        decoder = GRU(564, return_sequences=True)(decoder)
        decoder = GRU(564, return_sequences=False)(decoder)

        # decoder = LSTM(512, return_sequences=True)(decoder)
        # decoder = LSTM(512, return_sequences=False)(decoder)
        
        decoder = Dense(output_size, activation='softmax')(decoder)

        self.model = Model(inputs=[visual_input, textual_input], outputs=decoder)

        optimizer = RMSprop(lr=0.0001, clipvalue=1.0)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    def fit(self, images, partial_captions, next_words, output_path):

        filepath= output_path + "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

        self.model.fit([images, partial_captions], next_words, shuffle=False, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, callbacks=callbacks_list)
        self.save()

    def fit_generator(self, generator, output_path, steps_per_epoch ):
        filepath= output_path + "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]


        self.model.fit_generator(generator, steps_per_epoch=steps_per_epoch, epochs=EPOCHS, verbose=1,  callbacks=callbacks_list)
        self.save()

    def predict(self, image, partial_caption):
        return self.model.predict([image, partial_caption], verbose=0)[0]

    def predict_batch(self, images, partial_captions):
        return self.model.predict([images, partial_captions], verbose=1)
