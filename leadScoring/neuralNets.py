from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.neural_network import MLPRegressor
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn import metrics
from keras.regularizers import l2, l1


class Keras(object):

    def __init__(self, loss, inputDim, outputDim, hlActivation='relu', olActivation='sigmoid',
                 initializer='random_normal', optimizer='adam', metrics='accuracy'):
        self.hlActivation = hlActivation
        self.olActivation = olActivation
        self.initializer = initializer
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.inputDim = inputDim
        self.outputDim = outputDim

    def modelBuild(self, nodes, X_Train, y_Train, X_Test, y_Test, checkpointFile,
                   batch_size=10, epochs=100, early_stopping_delta=1e-4,
                   early_stopping_patience=10, dropout=0.01, ker_reg=0.000001,
                   ker_reg_type='l1'):
        # create model
        if ker_reg_type == 'l1':
            kernel_regularizer = l1(ker_reg)
        else:
            kernel_regularizer = l2(ker_reg)

        model = Sequential()
        # hidden layers
        for h in range(len(nodes)):
            if h == 0:
                model.add(Dense(nodes[h], input_dim=self.inputDim, activation=self.hlActivation,
                          kernel_initializer=self.initializer))
            elif h < len(nodes) - 1:
                model.add(Dropout(dropout))
            elif h > 0:
                model.add(Dense(nodes[h], activation=self.hlActivation,
                                kernel_regularizer=kernel_regularizer))
        # Output Layer
        model.add(Dense(self.outputDim, activation=self.olActivation,
                        kernel_initializer=self.initializer))
        # Compile model
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=[self.metrics])
        # save best model
        monitor = EarlyStopping(monitor='val_loss', min_delta=early_stopping_delta,
                                patience=early_stopping_patience, verbose=1, mode='auto')
        checkpointer = ModelCheckpoint(filepath=checkpointFile, verbose=0, save_best_only=True)
        model.fit(X_Train, y_Train, batch_size=batch_size, epochs=epochs, verbose=2,
                  callbacks=[monitor, checkpointer], validation_data=(X_Test, y_Test))
        # load weights from best model
        model.load_weights(checkpointFile)
        return model