import tensorflow as tf
import mlflow
import tqdm
from helper import preprocess_x, np


class SudokuBreaker:
    def __init__(self, path=None, layers=None, average_pooling=False):
        self.params = {}
        self.train_loss = None
        self.val_loss = None
        self.train_acc = None
        self.val_acc = None
        self.train_masked_loss = None
        self.val_masked_loss = None
        self.train_masked_acc = None
        self.val_masked_acc = None
        self.batch = None
        self.epochs = None
        self.learning_rate = None
        self.custom_training = False
        if not path:
            self.layers = layers
            self.average_pooling = average_pooling
            self.model = self.build_architecture()
            self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
            self.params.update({'layers': layers, 'average_pooling': average_pooling})
        else:
            self.model = self.load(path)
        self.hist = {}

    def to_tensorflow_dataset(self, X, y, batch=32, shuffle=100, training=False):
        '''
        Cast data to tensorflow dataset.
        '''
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        if training:
            dataset = dataset.shuffle(shuffle).batch(batch)
        else:
            dataset = dataset.batch(batch)
        return dataset

    def build_architecture(self):
        '''
        Builds model architecture.
        '''
        tf.keras.backend.clear_session()
        input_layer = tf.keras.layers.Input(shape=(9, 9, 1), dtype=tf.float32)
        for c, i in enumerate(self.layers):
            if c == 0:
                x = tf.keras.layers.Conv2D(filters=i, kernel_size=(9, 9), activation='relu', padding='same')(
                    input_layer)
                x = tf.keras.layers.BatchNormalization()(x)
            else:
                x = tf.keras.layers.Conv2D(filters=i, kernel_size=(3, 3), activation='relu', padding='same')(x)
                x = tf.keras.layers.BatchNormalization()(x)
        if self.average_pooling:
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
        else:
            x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(81 * 9)(x)
        x = tf.keras.layers.Reshape((-1, 9))(x)
        output_layer = tf.keras.layers.Activation('softmax')(x)
        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        print(model.summary())
        return model

    def masked_loss(self, x, y_true, y_pred):
        idx = tf.reshape(tf.keras.backend.equal(x, 0), shape=(x.shape[0], 81, 1))
        idx_y_pred = tf.broadcast_to(idx, y_pred.shape)
        y_pred = tf.reshape(tf.boolean_mask(y_pred, idx_y_pred), shape=(-1, 9))
        y_true = tf.reshape(tf.boolean_mask(y_true, idx), shape=(-1, 1))
        loss = self.loss_object(y_true, y_pred)
        return loss

    def masked_acc(self, x, y_pred, y_true):
        idx = tf.reshape(tf.keras.backend.equal(x, 0), shape=(x.shape[0], 81, 1))
        y_pred = tf.boolean_mask(tf.expand_dims(tf.argmax(y_pred, axis=-1), axis=-1), idx)
        y_true = tf.boolean_mask(y_true, idx)
        equal_stuff = tf.cast(tf.equal(y_true, y_pred), tf.int32)
        acc = tf.keras.backend.sum(equal_stuff) / equal_stuff.shape[0]
        return acc

    def loss(self, x, y_true, training=True):
        y_pred = self.model(tf.cast(x, dtype=tf.float32), training)
        return self.loss_object(y_true=y_true, y_pred=y_pred)

    def get_train_loss(self):
        return self.train_loss

    def get_train_masked_loss(self):
        return self.train_masked_loss

    def get_train_acc(self):
        return self.train_acc

    def get_train_masked_acc(self):
        return self.train_masked_acc

    def get_val_loss(self):
        return self.val_loss

    def get_val_masked_loss(self):
        return self.val_masked_loss

    def get_val_acc(self):
        return self.val_acc

    def get_val_masked_acc(self):
        return self.val_masked_acc

    def get_params(self):
        return self.get_params()

    def __predict(self, x, training=True):
        x = tf.cast(x, dtype=tf.float32)
        return self.model(x, training=training)

    def grad(self, X, y, training=True):
        with tf.GradientTape() as tape:
            loss_value = self.loss(X, y, training)
        return loss_value, tape.gradient(loss_value, self.model.trainable_variables)

    def fit_inbuilt(self, train_x, train_y, test_x, test_y, batch=32, epochs=10, learning_rate=0.001):
        '''
        Trains the model.
        train_x, train_y - train features and labels
        test_x, test_y - test features and labels
        batch - batch size that will be used for training
        epochs - number of epochs
        learning_rate - learning rate that will be used for training
        '''
        self.batch = batch
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.custom_training = False
        self.params.update({'batch': self.batch,
                            'epochs': self.epochs,
                            'learning_rate': self.learning_rate,
                            'custom_training': self.custom_training})
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                           loss='sparse_categorical_crossentropy', metrics=['acc'])

        hist = self.model.fit(train_x, train_y, validation_data=[test_x, test_y], epochs=epochs,
                              batch_size=batch)
        self.hist = hist.history
        self.train_loss = self.hist['loss'][-1]
        self.val_loss = self.hist['val_loss'][-1]
        self.train_acc = self.hist['acc'][-1]
        self.val_acc = self.hist['val_acc'][-1]

    def save(self, path):
        '''
        Saves trained model.
        path - path to the model to save it
        '''
        self.model.save(path)

    def load(self, path):
        '''
        Loads trained model.
        path - path to the model to save it
        '''
        return tf.keras.models.load_model(path, compile=False)

    # TODO: complete the function
    def predict_on_batch(self):
        pass

    def predict(self, x):
        '''
        Given sample x -> predict y
        '''
        if isinstance(x, str):
            feat = preprocess_x(x)
        else:
            feat = x
        while True:
            out = self.model.predict(feat.reshape((1, 9, 9, 1)))
            out = out.squeeze()

            pred = np.argmax(out, axis=1).reshape((9, 9)) + 1
            prob = np.around(np.max(out, axis=1).reshape((9, 9)), 2)

            feat = feat * 9
            feat = feat.reshape((9, 9))
            mask = (feat == 0)

            if mask.sum() == 0:
                break

            prob_new = prob * mask

            ind = np.argmax(prob_new)
            x, y = (ind // 9), (ind % 9)

            val = pred[x][y]
            feat[x][y] = val
            feat = feat / 9
        return pred

    def eval(self, test_x, test_y):
        result = np.apply_along_axis(arr=test_x, axis=1, func1d=self.predict)
        return np.equal(result, test_y).astype(np.int32).mean()

    def fit_custom(self, train_x, train_y, test_x, test_y, batch=32, epochs=10, learning_rate=0.001,
                   id_mlflow=25):
        '''
        Trains the model with respect to the custom masked_loss and custom masked_accuracy.
        (By default it's slower than fit_inbuilt function)
        train_x, train_y - train features and labels
        test_x, test_y - test features and labels
        batch - batch size that will be used for training
        epochs - number of epochs
        learning_rate - learning rate that will be used for training
        '''
        with mlflow.start_run(run_name=str(id_mlflow)):
            train_dataset = self.to_tensorflow_dataset(train_x, train_y, batch=batch, training=True)
            if test_x.size > 0:
                test_dataset = self.to_tensorflow_dataset(test_x, test_y, batch=batch, training=False)
            else:
                test_dataset = None
            self.batch = batch
            self.epochs = epochs
            self.learning_rate = learning_rate
            self.custom_training = True
            self.params.update({'batch': self.batch,
                                'epochs': self.epochs,
                                'learning_rate': self.learning_rate,
                                'custom_training': self.custom_training})
            mlflow.log_params(self.params)
            train_loss_results = []
            train_accuracy_results = []
            train_masked_loss_results = []
            train_masked_accuracy_results = []
            test_loss_results = []
            test_accuracy_results = []
            test_masked_loss_results = []
            test_masked_accuracy_results = []
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

            for epoch in range(epochs):
                epoch_masked_loss_avg = tf.keras.metrics.Mean()
                epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
                epoch_masked_accuracy = tf.keras.metrics.Mean()
                epoch_loss_avg = tf.keras.metrics.Mean()
                epoch_test_masked_loss_avg = tf.keras.metrics.Mean()
                epoch_test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
                epoch_test_masked_accuracy = tf.keras.metrics.Mean()
                epoch_test_loss_avg = tf.keras.metrics.Mean()
                for x, y in tqdm.tqdm(train_dataset):
                    loss_value, grads = self.grad(x, y)
                    optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                    y_pred = self.__predict(x)
                    epoch_loss_avg(loss_value)
                    epoch_masked_accuracy(self.masked_acc(x, y_pred, y))
                    epoch_masked_loss_avg(self.masked_loss(x=x, y_true=y, y_pred=y_pred))
                    epoch_accuracy(y, y_pred)
                loss_res = epoch_loss_avg.result().numpy()
                acc_res = epoch_accuracy.result().numpy()
                masked_loss_res = epoch_masked_loss_avg.result().numpy()
                masked_acc_res = epoch_masked_accuracy.result().numpy()
                mlflow.log_metric('train_loss', loss_res, step=epoch)
                mlflow.log_metric('train_acc', acc_res, step=epoch)
                mlflow.log_metric('train_masked_acc', masked_acc_res,step=epoch)
                mlflow.log_metric('train_masked_loss', masked_loss_res, step=epoch)
                train_accuracy_results.append(acc_res)
                train_masked_loss_results.append(masked_loss_res)
                train_masked_accuracy_results.append(masked_acc_res)
                train_loss_results.append(loss_res)
                print("Epoch {}: Masked Loss : {}, Masked accuracy: {},"
                      " Loss: {:.3f}, Accuracy: {}".format(epoch, masked_loss_res,
                                                           masked_acc_res,
                                                           loss_res,
                                                           acc_res))
                if test_dataset:
                    for x_test, y_test in test_dataset:
                        y_pred_test = self.__predict(x_test, training=False)
                        epoch_test_masked_loss_avg(self.masked_loss(x=x_test, y_true=y_test, y_pred=y_pred_test))
                        epoch_test_masked_accuracy(self.masked_acc(x_test, y_pred_test, y_test))
                        epoch_test_loss_avg(self.loss_object(y_true=y_test, y_pred=y_pred_test))
                        epoch_test_accuracy(y_test, y_pred_test)
                    test_loss_res = epoch_test_loss_avg.result().numpy()
                    test_acc_res = epoch_test_accuracy.result().numpy()
                    test_masked_loss_res = epoch_test_masked_loss_avg.result().numpy()
                    test_masked_acc_res = epoch_test_masked_accuracy.result().numpy()
                    mlflow.log_metric('val_loss', test_loss_res, step=epoch)
                    mlflow.log_metric('val_acc', test_acc_res, step=epoch)
                    mlflow.log_metric('val_masked_acc', test_masked_acc_res, step=epoch)
                    mlflow.log_metric('val_masked_loss', test_masked_loss_res, step=epoch)
                    test_loss_results.append(test_loss_res)
                    test_accuracy_results.append(test_acc_res)
                    test_masked_loss_results.append(test_masked_loss_res)
                    test_masked_accuracy_results.append(test_masked_acc_res)
                print("Validation masked Loss : {}, Validation masked accuracy: {},"
                      " Validation loss: {}, Validation accuracy: {}".format(test_masked_loss_res,
                                                                             test_masked_acc_res,
                                                                             test_loss_res,
                                                                             test_acc_res))
            self.hist = {'train_loss': train_loss_results,
                         'train_acc': train_accuracy_results,
                         'masked_train_loss': train_masked_loss_results,
                         'masked_train_acc': train_masked_accuracy_results,
                         'val_loss': test_loss_results,
                         'val_acc': test_accuracy_results,
                         'masked_val_loss': test_masked_loss_results,
                         'masked_val_acc': test_masked_accuracy_results,
                         }
            self.train_loss = train_loss_results[-1]
            self.val_loss = test_loss_results[-1]
            self.train_acc = train_accuracy_results[-1]
            self.val_acc = test_accuracy_results[-1]
            self.train_masked_loss = train_masked_loss_results[-1]
            self.val_masked_loss = test_masked_loss_results[-1]
            self.train_masked_acc = train_masked_accuracy_results[-1]
            self.val_masked_acc = test_masked_accuracy_results[-1]
