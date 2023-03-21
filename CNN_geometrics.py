import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Conv2D,
    Input,
    MaxPooling2D,
    Flatten,
    Dense,
    Concatenate,
    Dropout,
)

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from utils import *


class CustomLayer(tf.keras.layers.Layer):
    """Custom layer for CNN using geometric convolution"""

    def __init__(self, filters, kernel_size, padding="same", **kwargs):
        super(CustomLayer, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding

    def build(self, input_shape):
        self.conv2d = tf.keras.layers.Conv2D(
            self.filters, self.kernel_size, padding=self.padding
        )
        super(CustomLayer, self).build(input_shape)

    def call(self, inputs):
        x = tf.math.log(inputs)
        x = self.conv2d(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.math.exp(x)
        return x


def CNN_geom(input_shape, n_classes, lr):
    """CNN model using geometric convolution

    Parameters
    ----------
    input_shape : tuple
        tuple of input shape (height, width, channels)
    n_classes : int
        number of classes
    lr : float
        learning rate

    Returns
    -------
    tf.keras.models.Sequential
        CNN model
    """
    model = Sequential()
    # add convolutional layers
    model.add(CustomLayer(16, (3, 3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(CustomLayer(32, (4, 4)))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(CustomLayer(64, (5, 5)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(2048, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation="softmax"))
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    if n_classes == 2:
        model.compile(
            loss="binary_crossentropy",
            optimizer=opt,
            metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()],
        )
    else:
        model.compile(
            loss="categorical_crossentropy",
            optimizer=opt,
            metrics=[
                tf.keras.metrics.TopKCategoricalAccuracy(),
                tf.keras.metrics.CategoricalAccuracy(),
            ],
        )
    return model


def CNN_arth(input_shape, n_classes, lr):
    """CNN model using arithmetic convolution

    Parameters
    ----------
    input_shape : tuple
        tuple of input shape (height, width, channels)
    n_classes : int
        number of classes
    lr : float
        learning rate

    Returns
    -------
    tf.keras.models.Sequential
        CNN model
    """
    model = Sequential()
    model.add(
        Conv2D(16, (3, 3), activation="relu", padding="same", input_shape=input_shape)
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (4, 4), activation="relu", padding="same"))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Conv2D(64, (5, 5), activation="relu", padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(2048, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation="softmax"))
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    if n_classes == 2:
        model.compile(
            loss="binary_crossentropy",
            optimizer=opt,
            metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()],
        )
    else:
        model.compile(
            loss="categorical_crossentropy",
            optimizer=opt,
            metrics=[
                tf.keras.metrics.TopKCategoricalAccuracy(),
                tf.keras.metrics.CategoricalAccuracy(),
            ],
        )
    return model


def CNN_geom_arth(input_shape, n_classes, lr):
    """CNN model using geometric and arithmetic convolution

    Parameters
    ----------
    input_shape : tuple
        tuple of input shape (height, width, channels)
    n_classes : int
        number of classes
    lr : float
        learning rate

    Returns
    -------
    tf.keras.models.Model
        CNN model
    """
    input1 = Input(
        shape=input_shape,
    )
    b1 = Conv2D(16, (3, 3), activation="relu", padding="same")(input1)
    b1 = MaxPooling2D(pool_size=(2, 2))(b1)
    b1 = Conv2D(32, (3, 3), activation="relu", padding="same")(b1)
    b1 = MaxPooling2D(pool_size=(3, 3))(b1)
    b1 = Conv2D(32, (5, 5), activation="relu", padding="same")(b1)
    b1 = MaxPooling2D(pool_size=(4, 4))(b1)
    b1 = Flatten()(b1)

    b2 = CustomLayer(16, (3, 3))(input1)
    b2 = MaxPooling2D(pool_size=(2, 2))(b2)
    b2 = CustomLayer(32, (3, 3))(b2)
    b2 = MaxPooling2D(pool_size=(3, 3))(b2)
    b2 = CustomLayer(32, (5, 5))(b2)
    b2 = MaxPooling2D(pool_size=(4, 4))(b2)
    b2 = Flatten()(b2)
    merged = Concatenate()([b1, b2])
    x = Dense(2048, activation="relu")(merged)
    predictions = Dense(n_classes, activation="softmax")(x)
    model = Model(inputs=input1, outputs=predictions)

    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    if n_classes == 2:
        model.compile(
            loss="binary_crossentropy",
            optimizer=opt,
            metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()],
        )
    else:
        model.compile(
            loss="categorical_crossentropy",
            optimizer=opt,
            metrics=[
                tf.keras.metrics.TopKCategoricalAccuracy(),
                tf.keras.metrics.CategoricalAccuracy(),
            ],
        )
    return model


def mean_std_metrics(
    binary_metric, metric_types=["f1_score", "accuracy", "confusion_matrix"]
):
    results = {}
    for metric_type in metric_types:
        if metric_type == "f1_score":
            f1_scores = []
            for key in binary_metric:
                y_true = key["y_true"][:, 0]
                y_pred = key["y_est"][:, 0]
                y_pred = np.where(y_pred > 0.5, 1, 0)
                f1_scores.append(f1_score(y_true, y_pred))
            results["f1_score"] = np.array(f1_scores)
        elif metric_type == "accuracy":
            accuracies = []
            for key in binary_metric:
                y_true = key["y_true"][:, 0]
                y_pred = key["y_est"][:, 0]
                y_pred = np.where(y_pred > 0.5, 1, 0)
                accuracies.append(accuracy_score(y_true, y_pred))
            results["accuracy"] = np.array(accuracies)
        elif metric_type == "confusion_matrix":
            conf_matrices = []
            for key in binary_metric:
                y_true = key["y_true"][:, 0]
                y_pred = key["y_est"][:, 0]
                y_pred = np.where(y_pred > 0.5, 1, 0)
                conf_matrices.append(confusion_matrix(y_true, y_pred))
            results["conf_matrix"] = np.array(conf_matrices)
    return results


def metrics_from_binary_metric(binary_metric, save):
    arth_keys = [key for key in binary_metric.keys() if "CNN_arth_seed" in key]
    arth_dicts = [binary_metric[key] for key in arth_keys]

    geom_keys = [key for key in binary_metric.keys() if "CNN_geom_seed" in key]
    geom_dicts = [binary_metric[key] for key in geom_keys]

    arth_geom_keys = [
        key for key in binary_metric.keys() if "CNN_geom_arth_seed" in key
    ]
    arth_geom_dicts = [binary_metric[key] for key in arth_geom_keys]
    metrics = {
        "arth": mean_std_metrics(arth_dicts),
        "geom": mean_std_metrics(geom_dicts),
        "arth_geom": mean_std_metrics(arth_geom_dicts),
    }
    if save:
        dump_pkl(binary_metric, out_path + "binary_metric_bkcp.pkl")
        dump_pkl(metrics, out_path + "metrics.pkl")
    return metrics


def routine_limit_gpu(size):
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.set_visible_devices([], "GPU")
            tf.config.set_logical_device_configuration(
                gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=size)]
            )
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)


def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)


if __name__ == "__main__":

    routine_limit_gpu(1024)

    binary_metric, score = {}, {}
    lr = 5e-4
    input_shape = (16, 16, 1)
    n_classes = 2
    batch_size = 512
    epochs = 100

    m = {
        "CNN_geom": CNN_geom(input_shape, n_classes, lr),
        "CNN_arth": CNN_arth(input_shape, n_classes, lr),
        "CNN_geom_arth": CNN_geom_arth(input_shape, n_classes, lr),
    }

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=1e-7,
            patience=4,
            verbose=1,
        )
    ]

    data_path = "DATA/BINARY/"  # MULTI_LABELS/"
    out_path = "results/cnn_geom_bin/" + str(time.time()) + "/"
    os.makedirs(out_path, exist_ok=True)

    X_train, Y_train = open_pkl(data_path + "train_32.pkl")
    X_test, Y_test = open_pkl(data_path + "test_32.pkl")
    print(X_train.shape, X_test.shape)

    LabelEnc = LabelEncoder()
    y_train = LabelEnc.fit_transform(Y_train)
    y_test = LabelEnc.transform(Y_test)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    seeds = [826, 415]

    for keys in list(m.keys()):
        for seed in seeds:
            set_seed(seed)
            if keys == "CNN_geom":
                model = CNN_geom(input_shape, n_classes, lr)
            elif keys == "CNN_arth":
                model = CNN_arth(input_shape, n_classes, lr)
            elif keys == "CNN_geom_arth":
                model = CNN_geom_arth(input_shape, n_classes, lr)
            print(model.summary())
            history = model.fit(
                X_train,
                y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=2,
                workers=-1,
                use_multiprocessing=True,
            )
            dump_pkl(history.history, out_path + f"{keys}_seed_{seed}_history.pkl")
            print(f"-------------------{keys}----{seed}---------------")
            results = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=2)
            ypred = model.predict(X_test, batch_size=batch_size, verbose=2)
            print("================================================")
            print(keys, results)

            if n_classes == 2:
                score[keys + "_seed_" + str(seed)] = {
                    "loss": results[0],
                    "accuracy": results[1],
                    "auc": results[2],
                }
                binary_metric[keys + "_seed_" + str(seed)] = {"y_est": ypred}
                binary_metric[keys + "_seed_" + str(seed)]["y_true"] = y_test
            else:
                score[keys + "_seed_" + str(seed)] = {
                    "loss": results[0],
                    "topKaccuracy": results[1],
                    "category_accuracy": results[2],
                }

            tf.keras.backend.clear_session()
            del model
            dump_pkl(score, out_path + "score.pkl")

    for i in list(score.keys()):
        print(i, f": {score[i]:.4f}")

    if n_classes == 2:
        _ = metrics_from_binary_metric(binary_metric, save=True)

    print("=============== FINISHED ======================")
