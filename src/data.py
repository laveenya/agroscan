"""Dataset lists and the tf.data input pipeline (same logic as the Part 2 notebook)."""
import os

import tensorflow as tf

SETS = ["pv_train", "pv_val", "pw_train", "pw_val", "pw_test", "pd_train", "pd_val", "pd_test"]


def get_class_names(prepared_dir):
    root = os.path.join(prepared_dir, "pv_train")
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))


def items_of(prepared_dir, set_name, class_names):
    out = []
    for i, name in enumerate(class_names):
        folder = os.path.join(prepared_dir, set_name, name)
        out += [(os.path.join(folder, f), i) for f in sorted(os.listdir(folder)) if f.lower().endswith(".jpg")]
    return out


def load_all(prepared_dir):
    class_names = get_class_names(prepared_dir)
    return class_names, {s: items_of(prepared_dir, s, class_names) for s in SETS}


def read_img(path):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    return tf.cast(img, tf.float32)


def square_train(img, size):
    s = tf.reduce_min(tf.shape(img)[:2])
    img = tf.image.random_crop(img, tf.stack([s, s, 3]))
    return tf.image.resize(img, (size, size))


def square_eval(img, size):
    s = tf.reduce_min(tf.shape(img)[:2])
    img = tf.image.resize_with_crop_or_pad(img, s, s)
    return tf.image.resize(img, (size, size))


def make_ds(items, training, size, n_classes, seed):
    paths = [p for p, _ in items]
    labels = [l for _, l in items]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(items), seed=seed, reshuffle_each_iteration=True).repeat()
    fn = square_train if training else square_eval
    return ds.map(lambda p, y: (fn(read_img(p), size), tf.one_hot(y, n_classes)),
                  num_parallel_calls=tf.data.AUTOTUNE)


def eval_ds(items, params, n_classes, cache=False):
    d = params["data"]
    ds = make_ds(items, False, d["img_size"], n_classes, params["train"]["seed"]).batch(d["batch_size"])
    return (ds.cache() if cache else ds).prefetch(tf.data.AUTOTUNE)


def train_stream(data, params, n_classes):
    d, seed = params["data"], params["train"]["seed"]
    sources = {"pv": data["pv_train"], "pw": data["pw_train"], "pd": data["pd_train"]}
    dss, ws = [], []
    for key, items in sources.items():
        w = d["source_weights"].get(key, 0)
        if w > 0 and items:
            dss.append(make_ds(items, True, d["img_size"], n_classes, seed))
            ws.append(w)
    ws = [w / sum(ws) for w in ws]
    ds = tf.data.Dataset.sample_from_datasets(dss, weights=ws, seed=seed)
    return ds.batch(d["batch_size"]).prefetch(tf.data.AUTOTUNE)
