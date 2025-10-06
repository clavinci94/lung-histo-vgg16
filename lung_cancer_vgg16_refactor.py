"""
Refactored training script for lung histopathology classification
- Uses VGG16 (transfer learning) correctly
- Robust Grad-CAM utility
- Train/val/test split from a dataframe of image filepaths
- Class weights for imbalance
- Saves model and artifacts

Fill in DATA_DIR and run.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report

# -----------------------------
# Config
# -----------------------------
RANDOM_STATE = 42
IMAGE_SIZE = (224, 224)  # Recommended for VGG16
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-4
MODEL_DIR = "artifacts/model"
REPORTS_DIR = "artifacts/reports"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# -----------------------------
# Callbacks
# -----------------------------
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)
lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1)

# -----------------------------
# Data utilities
# -----------------------------

def prepare_data(data_dir: str) -> pd.DataFrame:
    """Scan class subfolders and build a dataframe with columns [filepaths, labels].

    Expected folder structure:
        data_dir/
            lung_aca/
            lung_scc/
            lung_n/
    """
    filepaths, labels = [], []

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    for cls in sorted(os.listdir(data_dir)):
        cls_path = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        for fname in os.listdir(cls_path):
            fpath = os.path.join(cls_path, fname)
            if os.path.isfile(fpath):
                filepaths.append(fpath)
                labels.append(cls)

    df = pd.DataFrame({"filepaths": filepaths, "labels": labels})
    df = df[df["filepaths"].apply(os.path.exists)].reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No images found. Check your DATA_DIR and subfolders.")
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1):
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=df["labels"],
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        random_state=RANDOM_STATE,
        stratify=train_val_df["labels"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def create_generators(train_df, val_df, test_df):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_dataframe(
        train_df,
        x_col="filepaths",
        y_col="labels",
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
        seed=RANDOM_STATE,
    )
    val_gen = val_test_datagen.flow_from_dataframe(
        val_df,
        x_col="filepaths",
        y_col="labels",
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )
    test_gen = val_test_datagen.flow_from_dataframe(
        test_df,
        x_col="filepaths",
        y_col="labels",
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )
    return train_gen, val_gen, test_gen

# -----------------------------
# Model
# -----------------------------

def create_model(input_shape, num_classes) -> Model:
    base_model = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=input_shape))
    base_model.trainable = False  # warm-up with frozen base

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# -----------------------------
# Training / Evaluation
# -----------------------------

def train_model(model: Model, train_gen, val_gen, epochs=EPOCHS, class_weights=None):
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        steps_per_epoch=len(train_gen),
        validation_steps=len(val_gen),
        class_weight=class_weights,
        callbacks=[early_stopping, lr_scheduler],
        verbose=1,
    )
    return history


def evaluate_model(model: Model, test_gen, report_prefix: str = "vgg16"):
    test_loss, test_acc = model.evaluate(test_gen, verbose=1)
    print(f"Test accuracy: {test_acc:.4f}")

    y_pred = np.argmax(model.predict(test_gen, verbose=1), axis=1)
    y_true = test_gen.classes
    labels = list(test_gen.class_indices.keys())

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    cm_path = os.path.join(REPORTS_DIR, f"{report_prefix}_confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=labels))

    return cm_path

# -----------------------------
# Grad-CAM
# -----------------------------

def grad_cam(model: Model, img_path: str, layer_name: str = "block5_conv3", out_path: str | None = None):
    """Compute Grad-CAM heatmap for a given image.

    layer_name should point to the last convolutional layer. For VGG16 this is 'block5_conv3'.
    """
    # Load & preprocess image
    img = tf.keras.utils.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_batch = np.expand_dims(img_array.copy(), axis=0)
    img_batch = preprocess_input(img_batch)

    # Build a model that maps the input image to the activations of the last conv layer and predictions
    last_conv_layer = model.get_layer(layer_name)
    grad_model = Model([model.inputs], [last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_batch)
        class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[0]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    max_val = heatmap.max() if heatmap.max() != 0 else 1e-9
    heatmap /= max_val

    # Superimpose
    original = cv2.imread(img_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    heatmap_resized = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)

    if out_path is None:
        out_path = os.path.join(REPORTS_DIR, "grad_cam_overlay.png")

    plt.figure(figsize=(8, 6))
    plt.imshow(overlay)
    plt.axis("off")
    plt.title("Grad-CAM Overlay")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return out_path

# -----------------------------
# Main workflow
# -----------------------------

def main():
    # 1) Set your dataset path here (macOS)
    DATA_DIR = "/Users/claudio/Desktop/lung_colon_image_set/lung_image_sets"  # <-- angepasst für macOS

    # 2) Build dataframe & splits
    df = prepare_data(DATA_DIR)
    train_df, val_df, test_df = split_data(df)

    # 3) Generators
    train_gen, val_gen, test_gen = create_generators(train_df, val_df, test_df)

    # 4) Class weights (map from class index -> weight)
    y_indices = train_gen.classes
    classes = np.unique(y_indices)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_indices)
    class_weights = {int(c): float(w) for c, w in zip(classes, weights)}

    # 5) Model
    input_shape = (*IMAGE_SIZE, 3)
    num_classes = len(train_gen.class_indices)
    model = create_model(input_shape, num_classes)

    # 6) Train (frozen base)
    history = train_model(model, train_gen, val_gen, epochs=EPOCHS, class_weights=class_weights)

    # 7) (Optional) Fine-tune: unfreeze last VGG16 block
    unfreeze = True
    if unfreeze:
        for layer in model.layers:
            if layer.name.startswith("block5_"):
                layer.trainable = True
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                      loss="categorical_crossentropy", metrics=["accuracy"])
        history_ft = train_model(model, train_gen, val_gen, epochs=5, class_weights=class_weights)

    # 8) Save model
    model_path = os.path.join(MODEL_DIR, "lung_vgg16.keras")
    model.save(model_path)

    # 9) Evaluate & confusion matrix
    cm_path = evaluate_model(model, test_gen, report_prefix="vgg16")

    print(f"Model saved to: {model_path}")
    print(f"Confusion matrix saved to: {cm_path}")

    # 10) Example Grad-CAM (set an existing image file path)
    # sample_img = test_df.iloc[0]["filepaths"]
    # cam_path = grad_cam(model, sample_img, layer_name="block5_conv3")
    # print(f"Grad-CAM saved to: {cam_path}")


if __name__ == "__main__":
    main()
