# 1. Imports

import os
import cv2
import gc
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    precision_score, recall_score, confusion_matrix,
    classification_report, roc_curve, auc, f1_score
)

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import VGG16, VGG19, ResNet50, ResNet152
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import load_model

# 2. Config

data_dir = r'C:\Users\Selesfia\OneDrive\Documents\Project2\Covid-19(2)\dataset2'
output_dir = "final_outputs_revised"
os.makedirs(output_dir, exist_ok=True)

image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
target_size = (128, 128)
num_classes = 2
batch_size = 8
epochs = 20
learning_rate = 0.0001
class_names = ['COVID', 'NonCOVID']
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

print("Dataset path exists:", os.path.exists(data_dir))
print("Output directory:", os.path.abspath(output_dir))

# 3. Load dataset

images = []
labels = []
count = 0

for folder in sorted(os.listdir(data_dir)):
    folder_path = os.path.join(data_dir, folder)

    if os.path.isdir(folder_path):
        for img_file in os.listdir(folder_path):
            if img_file.lower().endswith(image_extensions):
                img_path = os.path.join(folder_path, img_file)
                img = cv2.imread(img_path)

                if img is not None:
                    img = cv2.resize(img, target_size)
                    images.append(img)
                    labels.append(0 if folder == 'COVID' else 1)

                    count += 1
                    if count % 1000 == 0:
                        print(f"Loaded {count} images...")

images = np.array(images, dtype=np.uint8)
labels = np.array(labels, dtype=np.uint8)

print("Images shape:", images.shape)
print("Labels shape:", labels.shape)
print("COVID count:", np.sum(labels == 0))
print("NonCOVID count:", np.sum(labels == 1))

# 4. Train / validation / test split

# 80% temp, 20% test
X_temp, X_test, y_temp, y_test = train_test_split(
    images,
    labels,
    test_size=0.20,
    random_state=SEED,
    stratify=labels
)

# From remaining 80%:
# 70% train, 10% validation overall
X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.125,
    random_state=SEED,
    stratify=y_temp
)

# Keep datasets as uint8 to save RAM
# Only keep test display copy if you need visualization/XAI
X_test_display = X_test.copy()

print("\nSplit shapes:")
print("X_train:", X_train.shape, X_train.dtype)
print("X_val  :", X_val.shape, X_val.dtype)
print("X_test :", X_test.shape, X_test.dtype)
print("y_train:", y_train.shape)
print("y_val  :", y_val.shape)
print("y_test :", y_test.shape)

print("\nClass distribution:")
print("Train - COVID:", np.sum(y_train == 0), "NonCOVID:", np.sum(y_train == 1))
print("Val   - COVID:", np.sum(y_val == 0), "NonCOVID:", np.sum(y_val == 1))
print("Test  - COVID:", np.sum(y_test == 0), "NonCOVID:", np.sum(y_test == 1))

del images, labels, X_temp, y_temp
gc.collect()

# 5. Class imbalance handling

classes = np.unique(y_train)
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train
)
class_weights = {int(cls): float(weight) for cls, weight in zip(classes, class_weights_array)}

print("\nClass weights:", class_weights)

# 6. Preprocessing helpers

def get_preprocess_function(model_name):
    model_name = model_name.lower()
    if model_name == 'vgg16':
        return vgg16_preprocess
    elif model_name == 'vgg19':
        return vgg19_preprocess
    elif model_name in ['resnet50', 'resnet152']:
        return resnet_preprocess
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

def preprocess_for_model(X, model_name):
    preprocess_fn = get_preprocess_function(model_name)
    X = X.astype(np.float32, copy=False)
    return preprocess_fn(X)

# 7. Build transfer model

def build_transfer_model(base_name='VGG16', input_shape=(128, 128, 3), trainable=False):
    if base_name == 'VGG16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_name == 'VGG19':
        base_model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_name == 'ResNet152':
        base_model = ResNet152(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError("Unsupported model name")

    base_model.trainable = trainable

    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(2, activation='softmax', kernel_regularizer=l2(0.0001))(x)

    model = Model(inputs, outputs, name=base_name)

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# 8. Create models

model_vgg16 = build_transfer_model('VGG16', input_shape=(128, 128, 3), trainable=False)
model_vgg19 = build_transfer_model('VGG19', input_shape=(128, 128, 3), trainable=False)
model_resnet50 = build_transfer_model('ResNet50', input_shape=(128, 128, 3), trainable=False)
model_resnet152 = build_transfer_model('ResNet152', input_shape=(128, 128, 3), trainable=False)

# 9. Training generator

def make_train_generator(X_train, y_train, batch_size, seed, model_name):
    preprocess_fn = get_preprocess_function(model_name)

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )

    return train_datagen.flow(
        X_train,
        y_train,
        batch_size=batch_size,
        shuffle=True,
        seed=seed
    )


# 10. Train model

def train_model(model, model_name, X_train, y_train, X_val, y_val, class_weights=None):
    save_path = os.path.join(output_dir, f'bestrevised_{model_name.lower()}_model.keras')

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=8,
        verbose=1,
        restore_best_weights=True
    )

    model_checkpoint = ModelCheckpoint(
        save_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1
    )

    train_generator = make_train_generator(
        X_train, y_train, batch_size=batch_size, seed=SEED, model_name=model_name
    )

    X_val_preprocessed = preprocess_for_model(X_val, model_name)

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=(X_val_preprocessed, y_val),
        callbacks=[early_stopping, model_checkpoint, lr_scheduler],
        class_weight=class_weights,
        verbose=1
    )

    best_model = load_model(save_path)

    # free some RAM
    del X_val_preprocessed
    gc.collect()

    return history, best_model

history_vgg16, best_vgg16_model = train_model(
    model_vgg16, 'vgg16', X_train, y_train, X_val, y_val, class_weights=class_weights
)

history_vgg19, best_vgg19_model = train_model(
    model_vgg19, 'vgg19', X_train, y_train, X_val, y_val, class_weights=class_weights
)

history_resnet50, best_resnet50_model = train_model(
    model_resnet50, 'resnet50', X_train, y_train, X_val, y_val, class_weights=class_weights
)

history_resnet152, best_resnet152_model = train_model(
    model_resnet152, 'resnet152', X_train, y_train, X_val, y_val, class_weights=class_weights
)

# 11. Plot training history

def plot_training_history(history, model_name):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_training_history(history_vgg16, "VGG16")
plot_training_history(history_vgg19, "VGG19")
plot_training_history(history_resnet50, "ResNet50")
plot_training_history(history_resnet152, "ResNet152")

# 12. Evaluation helpers

def specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        return np.nan
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp + 1e-8)

def evaluate_model(model, X_test, y_test, model_name):
    X_eval = preprocess_for_model(X_test, model_name)

    y_pred_proba = model.predict(X_eval, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    test_loss, test_accuracy = model.evaluate(X_eval, y_test, verbose=0)

    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    specificity = specificity_score(y_test, y_pred)

    y_true_covid = (y_test == 0).astype(int)
    y_score_covid = y_pred_proba[:, 0]

    fpr, tpr, _ = roc_curve(y_true_covid, y_score_covid)
    roc_auc = auc(fpr, tpr)

    print(f"\n=== {model_name} ===")
    print(f"Test Loss      : {test_loss:.4f}")
    print(f"Accuracy       : {test_accuracy:.4f}")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"F1-score       : {f1:.4f}")
    print(f"Specificity    : {specificity:.4f}")
    print(f"AUC (COVID+)   : {roc_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    return {
        'Model': model_name,
        'Loss': test_loss,
        'Accuracy': test_accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1,
        'Specificity': specificity,
        'AUC': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

# 13. Evaluate all models on untouched test set

results_vgg16 = evaluate_model(best_vgg16_model, X_test, y_test, 'vgg16')
results_vgg19 = evaluate_model(best_vgg19_model, X_test, y_test, 'vgg19')
results_resnet50 = evaluate_model(best_resnet50_model, X_test, y_test, 'resnet50')
results_resnet152 = evaluate_model(best_resnet152_model, X_test, y_test, 'resnet152')

# 14. Result table

results_df = pd.DataFrame([
    {k: v for k, v in results_vgg16.items() if k not in ['y_pred', 'y_pred_proba', 'fpr', 'tpr']},
    {k: v for k, v in results_vgg19.items() if k not in ['y_pred', 'y_pred_proba', 'fpr', 'tpr']},
    {k: v for k, v in results_resnet50.items() if k not in ['y_pred', 'y_pred_proba', 'fpr', 'tpr']},
    {k: v for k, v in results_resnet152.items() if k not in ['y_pred', 'y_pred_proba', 'fpr', 'tpr']}
]).round(4)

print("\nResult Summary:")
print(results_df)

# 15. Confusion matrix

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

plot_confusion_matrix(y_test, results_vgg16['y_pred'], 'VGG16')
plot_confusion_matrix(y_test, results_vgg19['y_pred'], 'VGG19')
plot_confusion_matrix(y_test, results_resnet50['y_pred'], 'ResNet50')
plot_confusion_matrix(y_test, results_resnet152['y_pred'], 'ResNet152')

# 16. ROC curves

def plot_roc_curve_all(results_list):
    plt.figure(figsize=(10, 7))

    for result in results_list:
        plt.plot(
            result['fpr'],
            result['tpr'],
            linewidth=2,
            label=f"{result['Model']} (AUC = {result['AUC']:.4f})"
        )

    plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (COVID as Positive Class)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_roc_curve_all([
    results_vgg16,
    results_vgg19,
    results_resnet50,
    results_resnet152
])

# 17. Prediction visualization

def visualize_predictions(model, model_name, X_data, X_display, y_true, num_samples=8):
    indices = random.sample(range(len(X_data)), num_samples)

    plt.figure(figsize=(12, 12))

    for i, idx in enumerate(indices):
        img_raw = X_data[idx]
        img_show = X_display[idx]
        true_label = y_true[idx]

        img_input = preprocess_for_model(
            np.expand_dims(img_raw.astype(np.float32), axis=0),
            model_name
        )

        pred = model.predict(img_input, verbose=0)
        pred_label = np.argmax(pred)

        caption_color = 'green' if pred_label == true_label else 'red'

        plt.subplot(4, 4, i + 1)
        plt.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(
            f'Idx:{idx}\nPred:{class_names[pred_label]}\nTrue:{class_names[true_label]}',
            color=caption_color,
            fontsize=9
        )

    plt.tight_layout()
    plt.show()

visualize_predictions(best_vgg16_model, 'vgg16', X_test, X_test_display, y_test)


# 18. XAI helpers

def normalize_heatmap(heatmap):
    heatmap = np.maximum(heatmap, 0)
    max_val = np.max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    return heatmap

def overlay_heatmap(heatmap, img, alpha=0.15):
    heatmap = normalize_heatmap(heatmap)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    img = img.astype(np.float32)
    heatmap_color = heatmap_color.astype(np.float32)

    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return np.uint8(np.clip(overlay, 0, 255))

def overlay_heatmap_thresholded(heatmap, img, alpha=0.15, threshold=0.45):
    heatmap = normalize_heatmap(heatmap)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.where(heatmap >= threshold, heatmap, 0)

    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    img = img.astype(np.float32)
    heatmap_color = heatmap_color.astype(np.float32)

    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return np.uint8(np.clip(overlay, 0, 255))

# 19. Backbone helpers

def get_backbone_model(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer
    raise ValueError("No nested backbone model found.")


def build_gradcam_model(model, target_layer_name):
    backbone = get_backbone_model(model)

    target_layer = backbone.get_layer(target_layer_name)

    # very important:
    # extract BOTH target conv output and backbone final output
    # from the SAME backbone graph
    backbone_multi_output = tf.keras.Model(
        inputs=backbone.input,
        outputs=[target_layer.output, backbone.output]
    )

    inputs = model.input
    conv_output, backbone_output = backbone_multi_output(inputs)

    # outer classifier head
    x = model.layers[-3](backbone_output)   # GlobalAveragePooling2D
    x = model.layers[-2](x)                 # Dropout
    outputs = model.layers[-1](x)           # Dense

    grad_model = tf.keras.Model(
        inputs=inputs,
        outputs=[conv_output, outputs]
    )

    return grad_model

# 20. Grad-CAM

def make_gradcam_heatmap(img_array, model, class_index=None, target_layer_name=None):
    if target_layer_name is None:
        raise ValueError("Please provide target_layer_name")

    grad_model = build_gradcam_model(model, target_layer_name)

    img_array = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)

        if class_index is None:
            class_index = tf.argmax(predictions[0])

        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        raise ValueError("Gradients are None. Conv outputs are not connected to model output.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()

# 21. Grad-CAM++

def make_gradcam_plus_plus(img_array, model, class_index=None, target_layer_name=None):
    if target_layer_name is None:
        raise ValueError("Please provide target_layer_name")

    grad_model = build_gradcam_model(model, target_layer_name)

    img_array = tf.cast(img_array, tf.float32)

    with tf.GradientTape(persistent=True) as tape3:
        with tf.GradientTape(persistent=True) as tape2:
            with tf.GradientTape(persistent=True) as tape1:
                conv_outputs, predictions = grad_model(img_array, training=False)

                tape1.watch(conv_outputs)
                tape2.watch(conv_outputs)
                tape3.watch(conv_outputs)

                if class_index is None:
                    class_index = tf.argmax(predictions[0])

                loss = predictions[:, class_index]

            first_grads = tape1.gradient(
                loss, conv_outputs,
                unconnected_gradients=tf.UnconnectedGradients.ZERO
            )

        second_grads = tape2.gradient(
            first_grads, conv_outputs,
            unconnected_gradients=tf.UnconnectedGradients.ZERO
        )

    third_grads = tape3.gradient(
        second_grads, conv_outputs,
        unconnected_gradients=tf.UnconnectedGradients.ZERO
    )

    conv_outputs = conv_outputs[0]
    first_grads = first_grads[0]
    second_grads = second_grads[0]
    third_grads = third_grads[0]

    global_sum = tf.reduce_sum(conv_outputs, axis=(0, 1))

    alpha_num = second_grads
    alpha_denom = 2.0 * second_grads + third_grads * global_sum
    alpha_denom = tf.where(
        tf.abs(alpha_denom) > 1e-10,
        alpha_denom,
        tf.ones_like(alpha_denom)
    )

    alphas = alpha_num / alpha_denom
    alpha_norm = tf.reduce_sum(alphas, axis=(0, 1), keepdims=True)
    alphas = alphas / (alpha_norm + 1e-8)

    weights = tf.reduce_sum(tf.nn.relu(first_grads) * alphas, axis=(0, 1))

    heatmap = tf.reduce_sum(conv_outputs * weights, axis=-1)
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    del tape1, tape2, tape3
    return heatmap.numpy()

# 22. XAI visualization

def show_cam_result(model, model_name, X_data, X_display, y_data=None, idx=0,
                    method="gradcam", class_index=None, target_layer_name=None,
                    alpha=0.15, thresholded=False, threshold=0.45):
    img_raw = X_data[idx].astype(np.float32)
    img_show = X_display[idx].copy()

    img_array = np.expand_dims(img_raw, axis=0)
    img_array_preprocessed = preprocess_for_model(img_array, model_name)

    pred = model.predict(img_array_preprocessed, verbose=0)
    pred_label = int(np.argmax(pred[0]))

    if class_index is None:
        class_index = pred_label

    if method.lower() == "gradcam++":
        heatmap = make_gradcam_plus_plus(
            img_array_preprocessed,
            model,
            class_index=class_index,
            target_layer_name=target_layer_name
        )
        title_method = "Grad-CAM++"
    else:
        heatmap = make_gradcam_heatmap(
            img_array_preprocessed,
            model,
            class_index=class_index,
            target_layer_name=target_layer_name
        )
        title_method = "Grad-CAM"

    if thresholded:
        overlay = overlay_heatmap_thresholded(
            heatmap, img_show, alpha=alpha, threshold=threshold
        )
    else:
        overlay = overlay_heatmap(
            heatmap, img_show, alpha=alpha
        )

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
    if y_data is not None:
        plt.title(f"Original\nTrue: {class_names[y_data[idx]]}")
    else:
        plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap="jet")
    plt.title(f"{title_method} Heatmap")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title(f"{title_method} Overlay\nPred: {class_names[pred_label]}")
    plt.axis("off")

    plt.suptitle(f"Target layer: {target_layer_name} | Explained class: {class_names[class_index]}")
    plt.tight_layout()
    plt.show()

def compare_gradcam_vs_gradcampp(model, model_name, X_data, X_display, y_data=None,
                                 idx=0, class_index=None, target_layer_name=None,
                                 alpha=0.15):
    img_raw = X_data[idx].astype(np.float32)
    img_show = X_display[idx].copy()

    img_array = np.expand_dims(img_raw, axis=0)
    img_array_preprocessed = preprocess_for_model(img_array, model_name)

    pred = model.predict(img_array_preprocessed, verbose=0)
    pred_label = int(np.argmax(pred[0]))

    if class_index is None:
        class_index = pred_label

    heatmap_gc = make_gradcam_heatmap(
        img_array_preprocessed, model,
        class_index=class_index,
        target_layer_name=target_layer_name
    )

    heatmap_gcpp = make_gradcam_plus_plus(
        img_array_preprocessed, model,
        class_index=class_index,
        target_layer_name=target_layer_name
    )

    overlay_gc = overlay_heatmap(heatmap_gc, img_show, alpha=alpha)
    overlay_gcpp = overlay_heatmap(heatmap_gcpp, img_show, alpha=alpha)

    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
    if y_data is not None:
        plt.title(f"Original\nTrue: {class_names[y_data[idx]]}")
    else:
        plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(cv2.cvtColor(overlay_gc, cv2.COLOR_BGR2RGB))
    plt.title("Grad-CAM")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(cv2.cvtColor(overlay_gcpp, cv2.COLOR_BGR2RGB))
    plt.title("Grad-CAM++")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(heatmap_gcpp, cmap="jet")
    plt.title("Grad-CAM++ Heatmap")
    plt.axis("off")

    plt.suptitle(
        f"Target layer: {target_layer_name} | Pred: {class_names[pred_label]} | Explained class: {class_names[class_index]}"
    )
    plt.tight_layout()
    plt.show()

# 23. Compare all models

models = {
    "VGG16": best_vgg16_model,
    "VGG19": best_vgg19_model,
    "ResNet50": best_resnet50_model,
    "ResNet152": best_resnet152_model
}

target_layers = {
    "VGG16": "block5_conv3",
    "VGG19": "block5_conv4",
    "ResNet50": "conv5_block3_out",
    "ResNet152": "conv5_block3_out"
}

def compare_cam_all_models(models_dict, X_data, X_display, y_data=None,
                           idx=0, method="gradcam", target_layers=None,
                           class_index=None, alpha=0.15):
    img_raw = X_data[idx].astype(np.float32)
    img_show = X_display[idx].copy()

    n_models = len(models_dict)
    plt.figure(figsize=(4 * (n_models + 1), 5))

    plt.subplot(1, n_models + 1, 1)
    plt.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
    if y_data is not None:
        plt.title(f"Original\nTrue: {class_names[y_data[idx]]}")
    else:
        plt.title("Original")
    plt.axis("off")

    title_method = "Grad-CAM++" if method.lower() == "gradcam++" else "Grad-CAM"

    for i, (name, model) in enumerate(models_dict.items(), start=2):
        try:
            layer_name = target_layers[name]

            img_array = np.expand_dims(img_raw, axis=0)
            img_array_preprocessed = preprocess_for_model(img_array, name.lower())

            pred = model.predict(img_array_preprocessed, verbose=0)
            pred_label = int(np.argmax(pred[0]))
            explain_class = pred_label if class_index is None else class_index

            if method.lower() == "gradcam++":
                heatmap = make_gradcam_plus_plus(
                    img_array_preprocessed,
                    model,
                    class_index=explain_class,
                    target_layer_name=layer_name
                )
            else:
                heatmap = make_gradcam_heatmap(
                    img_array_preprocessed,
                    model,
                    class_index=explain_class,
                    target_layer_name=layer_name
                )

            overlay = overlay_heatmap(heatmap, img_show, alpha=alpha)

            plt.subplot(1, n_models + 1, i)
            plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            plt.title(f"{name}\nPred: {class_names[pred_label]}")
            plt.axis("off")

        except Exception as e:
            plt.subplot(1, n_models + 1, i)
            plt.text(0.5, 0.5, str(e), ha='center', va='center', wrap=True)
            plt.title(name)
            plt.axis("off")

    plt.suptitle(f"{title_method} Comparison Across Models", fontsize=16)
    plt.tight_layout()
    plt.show()

# 24. Example runs

idx = 0

sample_input = preprocess_for_model(
    np.expand_dims(X_test[idx], axis=0),
    'vgg16'
)
pred = best_vgg16_model.predict(sample_input, verbose=0)

print("\nPrediction probabilities:", pred[0])
print("Predicted class:", class_names[np.argmax(pred[0])])
print("True class:", class_names[y_test[idx]])

show_cam_result(
    model=best_vgg16_model,
    model_name='vgg16',
    X_data=X_test,
    X_display=X_test_display,
    y_data=y_test,
    idx=0,
    method="gradcam",
    class_index=None,
    target_layer_name="block5_conv3",
    alpha=0.15
)

compare_gradcam_vs_gradcampp(
    model=best_vgg16_model,
    model_name='vgg16',
    X_data=X_test,
    X_display=X_test_display,
    y_data=y_test,
    idx=idx,
    class_index=None,
    target_layer_name="block5_conv3",
    alpha=0.15
)

compare_cam_all_models(
    models_dict=models,
    X_data=X_test,
    X_display=X_test_display,
    y_data=y_test,
    idx=idx,
    method="gradcam",
    target_layers=target_layers,
    class_index=None,
    alpha=0.15
)

print("\nAll outputs saved to:", os.path.abspath(output_dir))
print("Files include best models and result summaries.")
print("This version uses a proper train/validation/test protocol.")

# 25. Grad-CAM vs Grad-CAM++ for ALL models

print("\nRunning Grad-CAM / Grad-CAM++ comparison for all models...")

# choose image index
idx = 0   # change anytime

model_dict = {
    "VGG16": {
        "model": best_vgg16_model,
        "model_name": "vgg16",
        "layer": "block5_conv3"
    },
    "VGG19": {
        "model": best_vgg19_model,
        "model_name": "vgg19",
        "layer": "block5_conv4"
    },
    "ResNet50": {
        "model": best_resnet50_model,
        "model_name": "resnet50",
        "layer": "conv5_block3_out"
    },
    "ResNet152": {
        "model": best_resnet152_model,
        "model_name": "resnet152",
        "layer": "conv5_block3_out"
    }
}

for model_title, info in model_dict.items():

    print(f"\n==============================")
    print(f"XAI Comparison: {model_title}")
    print(f"==============================")

    try:
        compare_gradcam_vs_gradcampp(
            model=info["model"],
            model_name=info["model_name"],
            X_data=X_test,
            X_display=X_test_display,
            y_data=y_test,
            idx=idx,
            class_index=None,
            target_layer_name=info["layer"],
            alpha=0.15
        )

    except Exception as e:
        print(f"{model_title} failed:", e)