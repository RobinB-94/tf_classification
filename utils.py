import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import tempfile
import cv2
from PIL import Image
import matplotlib

from keras.models import load_model, Model
from sklearn.metrics import confusion_matrix, classification_report, multilabel_confusion_matrix

from tensorflow import keras

from configs import encode_class_label


decodeGenus = {value: key for key, value in encode_class_label["x_cropped_0pad_224px_rgb_95orMore.npy"].items()}
decodeSex = {value: key for key, value in encode_class_label["sex"].items()}
genusLen = len(decodeGenus)

class Test_Set_Evaluation:
    def __init__(
        self,
        model: Model,
        test_set: tf.data.Dataset,
        class_names: list[str],
        save_dir: str
    ):
        self._model = model
        self._test_set = test_set
        self._pred = []
        self._true_labels = []
        self._class_names = class_names
        self._save_dir = save_dir

        for x, y_true in test_set:
            y_pred = model.predict(x, verbose=0)
            y_pred = np.argmax(y_pred, axis=-1)
            y_true = np.argmax(y_true, axis=-1)

            self._pred.extend(y_pred)
            self._true_labels.extend(y_true)

        # confusion matrix in form "GENUS SEX", e.G. "Coptera male"
        self._cm = confusion_matrix(self._true_labels, self._pred)

        self._class_sums = np.sum(self._cm, axis=1, keepdims=True)
        self._class_counts = self._class_sums.flatten()

        self._loss, self._accuracy = model.evaluate(test_set, verbose=0, batch_size=32)
        print("Test performance best model:")
        print(f"Loss: {self._loss}")
        print(f"Accuracy: {self._accuracy}")

    def save_classification_report(self, file_name = 'classification_report.txt'):
        report = self.get_classification_report()
        report += f"\nLoss: {self._loss}\nAccuracy: {self._accuracy}"
        file_path = os.path.join(self._save_dir, file_name)

        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(report)

    def get_classification_report(self):
        return classification_report(self._true_labels, self._pred, target_names=self._class_names)

    def save_confusion_matrix(self, file_name: str, formatting: str):
        """
            'formatting' is 'd' for int or '.1%' for % with one decimal. 
        """
        file_path = os.path.join(self._save_dir, file_name)
        self._build_conf_mat_fig(formatting)
        plt.savefig(file_path)

    def plot_confusion_matrix(self, formatting: str):
        """
            'formatting' is 'd' for int or '.1%' for % with one decimal. 
        """
        self._build_conf_mat_fig(formatting)
        plt.show()

    def _build_conf_mat_fig(self, formatting: str):
        cm = self._cm
        if formatting == '.1%':
            cm = cm / self._class_sums

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            cmap='Blues',
            fmt=formatting,
            cbar=False,
            xticklabels=self._class_names,
            yticklabels=self._class_names
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')

        for i in range(len(self._class_counts)):
            plt.text(len(self._class_names), i + 0.5, f"N={self._class_counts[i]}", va='center')

class Test_Set_Evaluation_ML:
    def __init__(
        self,
        model: Model,
        test_set: tf.data.Dataset,
        save_dir: str
    ):
        self._model = model
        self._test_set = test_set
        self._pred = []
        self._true_labels = []
        self._save_dir = save_dir

        for x, y_true in test_set:
            y_pred = model.predict(x, verbose=0)
            # Index of max predicted genus
            max_index_genus_pred = np.argmax(y_pred[:, :genusLen], axis=1)
            max_index_genus_true = np.argmax(y_true[:, :genusLen], axis=1)
            # Index of max predicted sex
            max_index_sex_pred = np.argmax(y_pred[:, genusLen:], axis=1)
            max_index_sex_true = np.argmax(y_true[:, genusLen:], axis=1)

            y_pred_strings = [
                f"{decodeGenus[genusIdx]} {decodeSex[sexIdx]}"
                for genusIdx, sexIdx
                in zip(max_index_genus_pred, max_index_sex_pred)
            ]

            y_true_strings = [
                f"{decodeGenus[genusIdx]} {decodeSex[sexIdx]}"
                for genusIdx, sexIdx
                in zip(max_index_genus_true, max_index_sex_true)
            ]

            self._pred.extend(y_pred_strings)
            self._true_labels.extend(y_true_strings)

        # confusion matrix in form "GENUS SEX", e.G. "Coptera male"
        # self._cm = multilabel_confusion_matrix(self._true_labels, self._pred)
        self._cm = confusion_matrix(self._true_labels, self._pred)

        # confusion matrix in form "GENUS", e.G. "Coptera"
        self._true_genus_labels = [label.split(" ")[0] for label in self._true_labels]
        self._pred_genus = [label.split(" ")[0] for label in self._pred]
        self._cm_genus = confusion_matrix(self._true_genus_labels, self._pred_genus)

        self._class_sums = np.sum(self._cm, axis=1, keepdims=True)
        self._class_sums_genus = np.sum(self._cm_genus, axis=1, keepdims=True)
        self._class_counts = self._class_sums.flatten()
        self._class_counts_genus = self._class_sums_genus.flatten()

        self._loss, self._accuracy = model.evaluate(test_set, verbose=0, batch_size=32)
        print("Test performance best model:")
        print(f"Loss: {self._loss}")
        print(f"Accuracy: {self._accuracy}")

    def plot_confs(self):
        for i, cm_label in enumerate(self._cm):
            sns.heatmap(cm_label, annot=True, fmt="d", cmap="Blues",
                        xticklabels=['Negativ', 'Positiv'],
                        yticklabels=['Negativ', 'Positiv'])
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title(f'Confusion Matrix - {self._class_names[i]}')
            plt.show()

    def save_classification_report(self, file_name = 'classification_report'):
        report = self.get_classification_report()
        report_genus = self.get_classification_report_genus_only()
        report += f"\nLoss: {self._loss}"
        file_path = os.path.join(self._save_dir, file_name + ".txt")
        file_path_genus = os.path.join(self._save_dir, file_name + "_genus.txt")

        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(report)
        with open(file_path_genus, 'w', encoding='utf-8') as file:
            file.write(report_genus)

    def get_classification_report(self):
        return classification_report(self._true_labels, self._pred)
    
    def get_classification_report_genus_only(self):
        return classification_report(self._true_genus_labels, self._pred_genus)

    def save_confusion_matrix(self, file_name: str, formatting: str):
        """
            'formatting' is 'd' for int or '.1%' for % with one decimal. 
        """
        file_path = os.path.join(self._save_dir, file_name + ".png")
        file_path_genus = os.path.join(self._save_dir, file_name + "_genus.png")
        self._build_conf_mat_fig(formatting)
        plt.savefig(file_path, bbox_inches='tight')
        self._build_conf_mat_genus_fig(formatting)
        plt.savefig(file_path_genus, bbox_inches='tight')

    def plot_confusion_matrix(self, formatting: str):
        """
            'formatting' is 'd' for int or '.1%' for % with one decimal. 
        """
        self._build_conf_mat_fig(formatting)
        plt.show()

    def _build_conf_mat_fig(self, formatting: str):
        cm = self._cm
        if formatting == '.1%':
            cm = cm / self._class_sums

        class_labels = sorted(list(set(self._true_labels)))

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            cmap='Blues',
            fmt=formatting,
            cbar=False,
            xticklabels=class_labels,
            yticklabels=class_labels
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.xticks(rotation=45, ha='right')

        for i in range(len(self._class_counts)):
            plt.text(2*genusLen, i + 0.5, f"N={self._class_counts[i]}", va='center')

    def _build_conf_mat_genus_fig(self, formatting: str):
        cm = self._cm_genus
        if formatting == '.1%':
            cm = cm / self._class_sums_genus

        class_labels = sorted(list(set(self._true_genus_labels)))

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            cmap='Blues',
            fmt=formatting,
            cbar=False,
            xticklabels=class_labels,
            yticklabels=class_labels
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.xticks(rotation=45, ha='right')

        for i in range(len(self._class_counts_genus)):
            plt.text(genusLen, i + 0.5, f"N={self._class_counts_genus[i]}", va='center')

def save_class_distribution(class_count_dict: dict, dir_path: str, class_names=[]):
    # Extract class names and number of images per class
    classes = list(class_count_dict.keys())
    counts = list(class_count_dict.values())

    if len(class_names):
        assert len(class_names) == len(classes)
        classes = class_names

    plt.figure()
    plt.bar(classes, counts)
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.title('Distribution of Number of Images per Class')
    plt.xticks(rotation=45)  # Rotates the x-axis labels by 45 degrees for better readability
    plt.tight_layout()
    plt.savefig(os.path.join(dir_path, 'class_distribution.png'))

def save_acc_loss_history(hist, dir_path):
    # save the loss
    plt.figure()
    plt.plot(hist.history['loss'], label='train loss')
    plt.plot(hist.history['val_loss'], label='val loss')
    plt.legend()
    plt.savefig(os.path.join(dir_path, 'LossVal_loss.png'))
    # plt.show()

    # save the accuracy
    plt.figure()
    plt.plot(hist.history['categorical_accuracy'], label='train acc')
    plt.plot(hist.history['val_categorical_accuracy'], label='val acc')
    plt.legend()
    plt.savefig(os.path.join(dir_path, 'AccVal_acc.png'))
    # plt.show()

def add_regularization(model: Model,
                       regularizer=tf.keras.regularizers.l2(1e-05),
                       reg_attributes=None,
                       custom_objects=None) -> Model:

    if not isinstance(regularizer, tf.keras.regularizers.Regularizer):
        raise ValueError('Regularizer must be a subclass of tf.keras.regularizers.Regularizer')
    
    if not reg_attributes:
        reg_attributes = ['kernel_regularizer', 'bias_regularizer',
                          'beta_regularizer', 'gamma_regularizer']

    for layer in model.layers:
        for attr in reg_attributes:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)

    # When we change the layers attributes, the change only happens in the model config file
    model_json = model.to_json()

    # Save the weights before reloading the model.
    tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
    model.save_weights(tmp_weights_path)

    # load the model from the config
    model = tf.keras.models.model_from_json(model_json, custom_objects=custom_objects)
    
    # Reload the model weights
    model.load_weights(tmp_weights_path, by_name=True)

    return model

def reshape(img: np.ndarray, output_shape: tuple, padding = 'zero_padding') -> np.ndarray:
    if padding == 'zero_padding':
        desired_size = output_shape[0]
        old_size = img.shape[:2] # old_size is in (height, width) format

        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size]) # new_size should be in (width, height) format

        img = cv2.resize(img, (new_size[1], new_size[0]))

        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        random_pos_w = np.random.randint(0, delta_w + 1)
        random_pos_h = np.random.randint(0, delta_h + 1)
        # random_pos_w = delta_w
        # random_pos_h = delta_h
        # top, bottom = delta_h//2, delta_h-top
        # left, right = delta_w//2, delta_w-left
        top, bottom = random_pos_h, delta_h-random_pos_h
        left, right = random_pos_w, delta_w-random_pos_w

        color = [0, 0, 0]
        new_im = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        # cv2.imshow("image", new_im)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return new_im
    
def undersample_class(
    images: np.ndarray,
    labels: np.ndarray,
    class_to_remove: int,
    imgs_to_retain: int
) -> tuple[np.ndarray, np.ndarray]:
    
    np.random.seed(0)
    # Indices of the images belonging to the class to be removed
    class_indices = np.where(labels == class_to_remove)[0]

    if len(class_indices) <= imgs_to_retain or imgs_to_retain == 0:
        return images, labels
    
    # Randomly choose images to remove
    remove_indices = np.random.choice(class_indices, size=len(class_indices) - imgs_to_retain, replace=False)

    # Remove the images from the dataset
    images = np.delete(images, remove_indices, axis=0)
    labels = np.delete(labels, remove_indices)

    # # Check the results
    # unique_labels, counts = np.unique(labels, return_counts=True)
    # print("Classes after removing images:", unique_labels)
    # print("Number of images per class:", counts)

    return images, labels


# https://keras.io/examples/vision/grad_cam/
class GradCAM:
    def __init__(self, model, last_conv_layer_name=None):
        # store the model, the class index used to measure the class
        # activation map, and the layer to be used when visualizing
        # the class activation map
        self.model = model
        self.layerName = last_conv_layer_name
        # if the layer name is None, attempt to automatically find
        # the target output layer
        if self.layerName is None:
            self.layerName = self._find_target_layer()
        # create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        self.grad_model = keras.models.Model(
            self.model.inputs,
            [self.model.get_layer(self.layerName).output, self.model.output]
        )

    def _find_target_layer(self):
        # attempt to find the final convolutional layer in the network
        # by looping over the layers of the network in reverse order
        for layer in reversed(self.model.layers):
            # check to see if the layer has a 4D output
            if len(layer.output_shape) == 4:
                return layer.name
        # otherwise, we could not find a 4D layer so the GradCAM
        # algorithm cannot be applied
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    def make_gradcam_heatmap(self, img_array, pred_index=None) -> np.ndarray:
        """
        Creates a 10x10 heatmap for given image(s).

        Args:
            img_array:
                shape: (number_of_images, img_shape).
                For example (100, 224, 224, 3).
            pred_index (optional):
                Gives heatmap for a specific output class. Defaults to None.
                If None: create heatmap for predicted class

        Returns:
            np.ndarray:
                shape: (number_of_images, heatmap_shape)
        """
        img_array = tf.cast(img_array, tf.float32)
        grad_model = self.grad_model

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            tape.watch(img_array)
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                # pred_index = tf.argmax(preds[0])
                pred_index = tf.argmax(preds, axis=1)
            # class_channel = preds[:, pred_index]
            class_channel = tf.gather(preds, pred_index, axis=1, batch_dims=0)

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        heatmap = tf.matmul(last_conv_layer_output, pooled_grads[:, tf.newaxis, :, tf.newaxis])
        heatmap = tf.squeeze(heatmap)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

    def save_and_display_gradcam(self,
                                 img,
                                 heatmap,
                                 cam_path="overlay.jpg",
                                 alpha=0.5,
                                 show_img=False,
                                 img_shape=None) -> np.ndarray:
        
        # Load the original image
        img = keras.utils.array_to_img(img)

        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)

        # Use jet colormap to colorize heatmap
        jet = matplotlib.colormaps["jet"]

        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # Create an image with RGB colorized heatmap
        jet_heatmap = keras.utils.array_to_img(jet_heatmap)

        if img_shape:
            jet_heatmap = jet_heatmap.resize(img_shape)
            img = img.resize(img_shape)
        else:
            jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))

        jet_heatmap = keras.utils.img_to_array(jet_heatmap)
        img = keras.utils.img_to_array(img)

        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * alpha + img

        # Save the superimposed image
        keras.utils.array_to_img(superimposed_img).save(cam_path)

        # Display Grad CAM
        if show_img:
            image = Image.open(cam_path)
            image.show()

        return superimposed_img
    


class Test_Set_Evaluation_MO:
    def __init__(
        self,
        model: Model,
        test_set: tf.data.Dataset,
        save_dir: str
    ):
        self._model = model
        self._test_set = test_set
        self._pred = []
        self._true_labels = []
        self._save_dir = save_dir

        for x, y_true in test_set:
            true_genus = np.argmax(y_true["genus"], axis=1)
            true_sex = np.array(y_true["sex"])

            y_pred = model.predict(x, verbose=0)

            pred_genus = np.argmax(y_pred[0], axis=1)
            pred_sex = np.where(y_pred[1] >= 0.5, 1, 0).flatten()

            y_true_strings = [
                f"{decodeGenus[genusIdx]} {decodeSex[sexIdx]}"
                for genusIdx, sexIdx
                in zip(true_genus, true_sex)
            ]

            y_pred_strings = [
                f"{decodeGenus[genusIdx]} {decodeSex[sexIdx]}"
                for genusIdx, sexIdx
                in zip(pred_genus, pred_sex)
            ]

            self._pred.extend(y_pred_strings)
            self._true_labels.extend(y_true_strings)

        # confusion matrix in form "GENUS SEX", e.G. "Coptera male"
        # self._cm = multilabel_confusion_matrix(self._true_labels, self._pred)
        self._cm = confusion_matrix(self._true_labels, self._pred)

        # confusion matrix in form "GENUS", e.G. "Coptera"
        self._true_genus_labels = [label.split(" ")[0] for label in self._true_labels]
        self._pred_genus = [label.split(" ")[0] for label in self._pred]
        self._cm_genus = confusion_matrix(self._true_genus_labels, self._pred_genus)

        self._class_sums = np.sum(self._cm, axis=1, keepdims=True)
        self._class_sums_genus = np.sum(self._cm_genus, axis=1, keepdims=True)
        self._class_counts = self._class_sums.flatten()
        self._class_counts_genus = self._class_sums_genus.flatten()

        _, _, _, self._acc_genus, self._acc_sex = model.evaluate(test_set, verbose=0, batch_size=32)
        print("Test performance best model:")
        print(f"Genus accuracy: {self._acc_genus}")
        print(f"Sex accuracy: {self._acc_sex}")

    def plot_confs(self):
        for i, cm_label in enumerate(self._cm):
            sns.heatmap(cm_label, annot=True, fmt="d", cmap="Blues",
                        xticklabels=['Negativ', 'Positiv'],
                        yticklabels=['Negativ', 'Positiv'])
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title(f'Confusion Matrix - {self._class_names[i]}')
            plt.show()

    def save_classification_report(self, file_name = 'classification_report'):
        report = self.get_classification_report()
        report_genus = self.get_classification_report_genus_only()
        report += f"\Genus accuracy: {self._acc_genus}"
        report += f"\nSex accuracy: {self._acc_sex}"
        file_path = os.path.join(self._save_dir, file_name + ".txt")
        file_path_genus = os.path.join(self._save_dir, file_name + "_genus.txt")

        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(report)
        with open(file_path_genus, 'w', encoding='utf-8') as file:
            file.write(report_genus)

    def get_classification_report(self):
        return classification_report(self._true_labels, self._pred)
    
    def get_classification_report_genus_only(self):
        return classification_report(self._true_genus_labels, self._pred_genus)

    def save_confusion_matrix(self, file_name: str, formatting: str):
        """
            'formatting' is 'd' for int or '.1%' for % with one decimal. 
        """
        file_path = os.path.join(self._save_dir, file_name + ".png")
        file_path_genus = os.path.join(self._save_dir, file_name + "_genus.png")
        self._build_conf_mat_fig(formatting)
        plt.savefig(file_path, bbox_inches='tight')
        self._build_conf_mat_genus_fig(formatting)
        plt.savefig(file_path_genus, bbox_inches='tight')

    def plot_confusion_matrix(self, formatting: str):
        """
            'formatting' is 'd' for int or '.1%' for % with one decimal. 
        """
        self._build_conf_mat_fig(formatting)
        plt.show()

    def _build_conf_mat_fig(self, formatting: str):
        cm = self._cm
        if formatting == '.1%':
            cm = cm / self._class_sums

        class_labels = sorted(list(set(self._true_labels)))

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            cmap='Blues',
            fmt=formatting,
            cbar=False,
            xticklabels=class_labels,
            yticklabels=class_labels
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.xticks(rotation=45, ha='right')

        for i in range(len(self._class_counts)):
            plt.text(2*genusLen, i + 0.5, f"N={self._class_counts[i]}", va='center')

    def _build_conf_mat_genus_fig(self, formatting: str):
        cm = self._cm_genus
        if formatting == '.1%':
            cm = cm / self._class_sums_genus

        class_labels = sorted(list(set(self._true_genus_labels)))

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            cmap='Blues',
            fmt=formatting,
            cbar=False,
            xticklabels=class_labels,
            yticklabels=class_labels
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.xticks(rotation=45, ha='right')

        for i in range(len(self._class_counts_genus)):
            plt.text(genusLen, i + 0.5, f"N={self._class_counts_genus[i]}", va='center')
