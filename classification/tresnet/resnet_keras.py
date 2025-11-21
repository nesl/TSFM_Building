# resnet model 
# when tuning start with learning rate->mini_batch_size -> 
# momentum-> #hidden_units -> # learning_rate_decay -> #layers 
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time
import keras_metrics as km

import matplotlib

import matplotlib.pyplot as plt

class ResNet:

    def __init__(self, output_directory, input_shape, nb_classes, patientID, verbose=False, build=True, load_weights=False, batch_size=8, nb_epochs=200, random_seed=13, max_lr=0.0001, n_feature_maps=64):
        
        self.n_feature_maps = n_feature_maps
        self.max_lr = max_lr
        self.output_directory = output_directory
        self.patientID = patientID
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        
        if build == True:
            self.model = self.build_model(self.input_shape, self.nb_classes)
            if (verbose == True):
                self.model.summary()
            self.verbose = verbose
            
            if load_weights == True:
                self.model.load_weights(self.output_directory + 'best_model_{}.hdf5'.format(self.patientID))
                print(f'Loaded model best_model_{self.patientID}.hdf5')
            # else:
            #     self.model.save_weights(self.output_directory + 'model_{}.hdf5'.format(self.patientID))
        return

    def build_model(self, input_shape, nb_classes):
        
        n_feature_maps = self.n_feature_maps

        input_layer = keras.layers.Input(input_shape)

        # BLOCK 1

        conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = keras.layers.Activation('relu')(output_block_1)

        # BLOCK 2

        conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_2 = keras.layers.add([shortcut_y, conv_z])
        output_block_2 = keras.layers.Activation('relu')(output_block_2)

        # BLOCK 3

        conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # no need to expand channels because they are equal
        shortcut_y = keras.layers.BatchNormalization()(output_block_2)

        output_block_3 = keras.layers.add([shortcut_y, conv_z])
        output_block_3 = keras.layers.Activation('relu')(output_block_3)

        # FINAL

        gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate = self.max_lr), metrics=['accuracy', tf.keras.metrics.Recall(name = 'recall')])

        # reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_recall', factor=0.5, patience=10, min_lr=0.00001, mode='max')

        es = keras.callbacks.EarlyStopping(monitor='val_recall', mode='max', min_delta = 0.05, patience = 10)

        file_path = self.output_directory + 'best_model_{}.hdf5'.format(self.patientID)

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_recall',
                                                           save_best_only=True, verbose = True, mode = 'max')
        
        # self.callbacks = [reduce_lr, model_checkpoint, es] 
        self.callbacks = [model_checkpoint, es]

        return model

    def fit(self, X_train, y_train, validation_split = None, X_val = None, y_val = None, 
            class_weight = None, sample_weight = None):   
        
        if not tf.test.is_gpu_available:
            print('error')
            exit()
        
        if ((X_val is None) and (validation_split is None)):
            raise ValueError('Atleast one of X_val or validation_split must be not none.')
        
        mini_batch_size = int(min(X_train.shape[0] / 10, self.batch_size))
        
        if X_val is not None:
            # x_val and y_val are only used to monitor the test loss and NOT for training
            hist = self.model.fit(X_train, y_train, batch_size=mini_batch_size, epochs=self.nb_epochs, 
                                  validation_data=(X_val, y_val), verbose=self.verbose, callbacks=self.callbacks, 
                                  class_weight=class_weight, sample_weight=sample_weight)
        
        elif validation_split is not None: 
            hist = self.model.fit(X_train, y_train, batch_size=mini_batch_size, epochs=self.nb_epochs, 
                                  validation_split=validation_split, verbose=self.verbose, callbacks=self.callbacks, 
                                  class_weight=class_weight, sample_weight=sample_weight)
        
        # keras.backend.clear_session()

    def predict(self, X_test, X_train, y_train, y_test):
        
        self.model = self.build_model(self.input_shape, self.nb_classes)
        
        self.model.load_weights(self.output_directory + 'best_model_{}.hdf5'.format(self.patientID))
       
        y_pred = self.model.predict(X_test)
        
        return y_pred