"""Model definition. Preprocessing and augmentation live INSIDE the model, so the API only sends raw 0-255 pixels."""
from tensorflow import keras
from tensorflow.keras import layers

BACKBONES = {
    "mobilenetv2":     dict(fn=keras.applications.MobileNetV2,     rescale=True),
    "efficientnetv2s": dict(fn=keras.applications.EfficientNetV2S, rescale=False),
    "convnext_tiny":   dict(fn=keras.applications.ConvNeXtTiny,    rescale=False),
}


def build_model(params, n_classes):
    size = params["data"]["img_size"]
    mp = params["model"]
    cfg = BACKBONES[mp["backbone"]]
    inputs = keras.Input((size, size, 3))
    x = inputs
    if mp["use_aug"]:
        x = keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomContrast(0.25),
            layers.RandomBrightness(0.25, value_range=(0, 255)),
        ], name="augment")(x)
    if cfg["rescale"]:
        x = layers.Rescaling(1 / 127.5, offset=-1, name="rescale")(x)
    base = cfg["fn"](input_shape=(size, size, 3), include_top=False, weights=mp["weights"])
    base.trainable = False
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(n_classes, dtype="float32")(x)
    outputs = layers.Activation("softmax", dtype="float32")(x)
    return keras.Model(inputs, outputs), base
