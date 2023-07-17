import os
import json
from collections import defaultdict
from configs import directory_hierarchy, encode_class_label
from utils import reshape, undersample_class

import numpy as np
import tensorflow as tf
from keras.layers.preprocessing.image_preprocessing import RandomRotation
from keras.layers.preprocessing.image_preprocessing import RandomTranslation
from keras.layers.preprocessing.image_preprocessing import RandomZoom
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

import cv2
from skimage import transform

np.random.seed(0)
tf.random.set_seed(0)

DATA_DIR = os.path.join(os.getcwd(), "images")
X_FILE_NAME = "x_cropped_0pad_224px_rgb_95orMore.npy"
X_FILE_PATH = os.path.join(DATA_DIR, X_FILE_NAME)
Y_GENUS_FILE_PATH = os.path.join(DATA_DIR, "y_genus_cropped_0pad_224px_rgb_95orMore.npy")
Y_SEX_FILE_PATH = os.path.join(DATA_DIR, "y_sex_cropped_0pad_224px_rgb_95orMore.npy")
IMG_CODES_FILE_PATH = os.path.join(DATA_DIR, "img-codes_cropped_0pad_224px_rgb_95orMore.npy")
DATA_FILE = os.path.join(DATA_DIR, "recrop", "cropped_imgs_merged.json")
IMG_SIZE = 224
IMG_DEPTH = 3
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, IMG_DEPTH)
dtype = np.uint8
# seed number for reproducable datasets
seed = 40


def extract_dataset(crop = False) -> None:
    with open(DATA_FILE, 'r') as f:
        data = json.load(f)

    data = dict(data)
    data = {key: value for key, value in data.items() if "exclude" not in value["comments"]}

    # get classes with at least 95 images only
    counts = defaultdict(int)
    for entry in data.values():
        gender = entry["genus"]
        counts[gender] += 1

    usable_classes = [key for key, value in counts.items() if value >= 95]
    usable_images = {key: item for key, item in data.items() if item["genus"] in usable_classes}
    num_images = sum(counts[key] for key in usable_classes)

    print(usable_classes)

    x = np.zeros(
        shape=(num_images, IMG_SIZE, IMG_SIZE, IMG_DEPTH), dtype=dtype
    )
    y = np.zeros(shape=(num_images,), dtype=np.uint8)
    img_codes = np.zeros(shape=(num_images,), dtype=object)

    cnt = 0
    error_msg = []
    for img_code, img_data in usable_images.items():
        sex = img_data["sex"]
        genus = img_data["genus"]
        img_name = img_data["file_name"]
        dir_path = directory_hierarchy[genus]
        img_file_path = os.path.join(DATA_DIR, *dir_path, sex, img_name)
        # crop
        crop = img_data["crop"]
        x0, x1, y0, y1 = crop["x0"], crop["x1"], crop["y0"], crop["y1"]

        try:
            img = cv2.imread(img_file_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if crop:
                img = img[y0:y1, x0:x1]
            x[cnt] = reshape(img=img, output_shape=IMG_SHAPE, padding="zero_padding")
            y[cnt] = encode_class_label[X_FILE_NAME][genus]
            img_codes[cnt] = img_code
            cnt += 1
            print(f"Image {cnt}/{num_images} done.")
        except KeyboardInterrupt:
            return
        except:
            error_msg.append(f"Image {img_file_path} can't be read!")

    for msg in error_msg:
        print(msg)

    # Dropping not readable image idxs
    x = x[:cnt]
    y = y[:cnt]
    img_codes = img_codes[:cnt]

    np.save(X_FILE_PATH, x)
    np.save(Y_GENUS_FILE_PATH, y)
    np.save(IMG_CODES_FILE_PATH, img_codes)

def create_sex_labels():
    # Load the image codes
    img_codes = np.load(IMG_CODES_FILE_PATH, allow_pickle=True)
    num_images = np.shape(img_codes)[0]
    y_sex = np.zeros(shape=(num_images,), dtype=np.uint8)

    with open('cropped_imgs.json', 'r') as f:
        data = json.load(f)
        data = dict(data)

    for idx, img_code in enumerate(img_codes):
        sex = data[img_code]["sex"]
        encoded_label = encode_class_label[X_FILE_NAME][sex]
        y_sex[idx] = encoded_label
        print(encoded_label)

    np.save(Y_SEX_FILE_PATH, y_sex)

class DATASET:
    def __init__(
        self,
        test_size: float = 0.2,
        validation_size: float = 0.15,
        class_balance = False,
        batch_size = 64,
    ) -> None:
        self.ordered_genera_classes = [*encode_class_label[X_FILE_NAME].keys()]
        self.ordered_sex = [*encode_class_label['sex'].keys()]
        self.ordered_all_classes = self.ordered_genera_classes + self.ordered_sex
        self.num_classes = len(self.ordered_all_classes)
        self.batch_size = batch_size
        # Load the data set
        x = np.load(X_FILE_PATH)
        y_genus = np.load(Y_GENUS_FILE_PATH)
        y_sex = np.load(Y_SEX_FILE_PATH)
        # img_codes = np.load(IMG_CODES_FILE_PATH, allow_pickle=False)
        # Preprocess y data
        y_genus = to_categorical(y_genus)
        y_sex = to_categorical(y_sex)
        y = np.concatenate((y_genus, y_sex), axis=1)
        # train/test-split
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            stratify=y if class_balance else None,
            random_state=seed
        )

        # train/val-split
        x_train, x_val, y_train, y_val = train_test_split(
            x_train,
            y_train,
            test_size=validation_size,
            stratify=y_train if class_balance else None,
            random_state=seed
        )

        # Preprocess x data
        self.x_train = x_train.astype(np.float32)
        self.x_test = x_test.astype(np.float32)
        self.x_val = x_val.astype(np.float32)
        # Preprocess y data
        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_val
        # Dataset attributes
        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]
        self.val_size = self.x_val.shape[0]
        self.width = self.x_train.shape[1]
        self.height = self.x_train.shape[2]
        self.depth = self.x_train.shape[3]
        self.img_shape = (self.width, self.height, self.depth)
        # tf.data Datasets
        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            (self.x_train, self.y_train)
        )
        self.test_dataset = tf.data.Dataset.from_tensor_slices(
            (self.x_test, self.y_test)
        )
        self.val_dataset = tf.data.Dataset.from_tensor_slices(
            (self.x_val, self.y_val)
        )
        # self.test_dataset.save("models/xxx/tf_test_set")
        self.train_dataset = self._prepare_dataset(
            self.train_dataset, shuffle=True, augment=False
        )
        self.test_dataset = self._prepare_dataset(self.test_dataset)
        self.val_dataset = self._prepare_dataset(self.val_dataset)

    def get_train_set(self) -> tf.data.Dataset:
        return self.train_dataset

    def get_test_set(self) -> tf.data.Dataset:
        return self.test_dataset

    def get_val_set(self) -> tf.data.Dataset:
        return self.val_dataset

    @staticmethod
    def load_and_preprocess_custom_image(image_file_path: str) -> np.ndarray:
        img = cv2.imread(image_file_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transform.resize(image=img, output_shape=IMG_SHAPE)
        return img

    @staticmethod
    def _build_data_augmentation() -> Sequential:
        model = Sequential()

        model.add(RandomRotation(factor=0.08))
        model.add(RandomTranslation(height_factor=0.08, width_factor=0.08))
        model.add(RandomZoom(height_factor=0.08, width_factor=0.08))

        return model

    def _prepare_dataset(
        self,
        dataset: tf.data.Dataset,
        shuffle: bool = False,
        augment: bool = False,
    ) -> tf.data.Dataset:
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1_000)

        dataset = dataset.batch(batch_size=self.batch_size)

        if augment:
            data_augmentation_model = self._build_data_augmentation()
            dataset = dataset.map(
                map_func=lambda x, y: (
                    data_augmentation_model(x, training=False),
                    y,
                ),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )

        return dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


if __name__ == "__main__":
    extract_dataset(crop=True)
    # create_sex_labels()
