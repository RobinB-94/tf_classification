import tensorflow as tf
from keras.models import Model, load_model
from keras.optimizers import Adam, SGD
# from keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from keras.applications.resnet import ResNet50, preprocess_input
# from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.layers import Flatten, Dense, Input, Lambda, Conv2D, MaxPooling2D, AveragePooling2D, Dropout, GlobalAvgPool2D, LayerNormalization, GlobalMaxPool2D
from keras.regularizers import L2, l1_l2
from keras.metrics import BinaryAccuracy, Precision, Recall, AUC

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os, shutil, time
from data_loader import DATASET
from configs import directory_hierarchy, encode_class_label
from utils import Test_Set_Evaluation, add_regularization, save_acc_loss_history, save_class_distribution, custom_loss, compute_batch_class_weights, gen, Test_Set_Evaluation_ML
import convnext


def create_model(input_shape,
                 n_classes,
                 optimizer,
                 fine_tune=0,
                 l1_reg=0.0,
                 l2_reg=0.0,
                 drop_path_rate=0.0):
    
    convnext_model = convnext.ConvNeXtBase(
        model_name="convnext_base",
        include_top=False,
        include_preprocessing=True,
        weights="imagenet",
        input_shape=input_shape,
        classifier_activation="sigmoid",
        drop_path_rate=drop_path_rate
    )

    if fine_tune:
        for layer in convnext_model.layers[:fine_tune]:
            layer.trainable = False
        for layer in convnext_model.layers[fine_tune:]:
            layer.trainable = True

    regularization = l1_l2(l1=l1_reg, l2=l2_reg)

    # Classification head: Global pooling + layer norm + fully connected layer
    x = GlobalMaxPool2D()(convnext_model.output)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Dense(units=n_classes, activation='sigmoid', kernel_regularizer=regularization)(x)

    model = Model(inputs=convnext_model.input, outputs=x)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[BinaryAccuracy()]
    )

    return model

if __name__ == "__main__":

    gpu_devices = tf.config.experimental.list_physical_devices("GPU")
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

    # Global params
    EPOCHS = 1000
    BATCH_SIZE = 50

    # apply resnet preprocessing
    with tf.device("CPU"):
        data = DATASET(
            test_size=0.2,
            validation_size=0.15,
            class_balance=True,
            batch_size=BATCH_SIZE,
            # undersample_imgs_to_retain=150
        )

    train_dataset = data.get_train_set()
    val_dataset = data.get_val_set()
    test_dataset = data.get_test_set()

    network_dir = f"dir_example_name"
    DATA_DIR = os.path.join(os.getcwd(), "models", network_dir)
    MODEL_LOG_DIR = os.path.join(os.getcwd(), "logs", network_dir)
    WEIGHTS_PATH = os.path.join(DATA_DIR, "x_0pad_224px_rgb_95orMore.weights.best.h5")

    if os.path.exists(DATA_DIR):
        raise Exception(f"The path {DATA_DIR} already exists.")
    else:
        os.mkdir(DATA_DIR)
    
    # copy python files to destination folder
    current_file = __file__
    data_loader_file = os.path.join(os.getcwd(), "data_loader.py")
    shutil.copy(current_file, DATA_DIR)
    shutil.copy(data_loader_file, DATA_DIR)

    img_shape = data.img_shape
    num_classes = data.num_classes

    # optimizer
    optimizer = Adam(learning_rate=0.001)

    convnext_model = create_model(img_shape,
                                    num_classes,
                                    optimizer,
                                    fine_tune=165,
                                    #l2_reg=0.01,
                                    # drop_path_rate=tune,
                                    # rev=rev,
                                    )

    # ModelCheckpoint callback - save best weights
    tl_checkpoint_1 = ModelCheckpoint(
        filepath=WEIGHTS_PATH,
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
        monitor='val_loss',
        mode='min'
    )

    # EarlyStopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
    )

    # lr_plateau = ReduceLROnPlateau(
    #     monitor="val_loss", factor=0.25, patience=5, verbose=1, min_lr=1e-6
    # )

    # tensorboard --logdir logs
    tb_callback = TensorBoard(
        log_dir=MODEL_LOG_DIR, histogram_freq=1, write_graph=True
    )

    hist = convnext_model.fit(
        train_dataset,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=[tl_checkpoint_1, early_stop, tb_callback],
        verbose=1,
    )

    evaluate = Test_Set_Evaluation_ML(
        convnext_model,
        test_dataset,
        DATA_DIR
    )

    evaluate.save_classification_report()
    evaluate.save_confusion_matrix('conf_mat_decimal', 'd')
    evaluate.save_confusion_matrix('conf_mat_perc', '.1%')
