# AgroScan - Trained model: metrics and results

**Model:** MobileNetV2 (ImageNet pretrained, fine-tuned), 2.6 million parameters
**Task:** classify a leaf photo into 8 crop-disease classes (tomato, potato, bell pepper)
**Data version:** `agroscan_prepared_v1` (PlantVillage + PlantWild + PlantDoc, 8 matching classes)
**Model file:** `AgroScan-v5/models/mobilenetv2.keras` (also saved as `agroscan_v5_mobilenetv2.keras`)

---

## 1. Headline results (held-out TEST photos, real-world images)

| Test set | Images | Top-1 accuracy | 95% interval | Top-3 accuracy | Macro-F1 |
|---|---|---|---|---|---|
| PlantDoc test (official test folder) | 72 | **62.5%** | 51.4 - 73.6% | 94.4% | 0.622 |
| PlantWild test (official test split) | 304 | **60.9%** | 55.3 - 66.4% | 88.5% | 0.601 |

- **Top-1:** the model's first answer is the correct disease.
- **Top-3:** the correct disease is among the model's three best guesses.
- With flip averaging (test-time augmentation) PlantDoc test is 63.9% top-1 (95% interval 52.8 - 75.0%), top-3 95.8%, macro-F1 0.638. This is one photo's difference and is not used as the headline.
- The test photos were never used for training, early stopping or model selection. Near-duplicates between training images and validation/test images were removed.

## 2. Accuracy when the model only answers if it is confident

| Min. confidence | PlantDoc test: photos answered | PlantDoc test: accuracy | PlantWild test: photos answered | PlantWild test: accuracy |
|---|---|---|---|---|
| 0.0 (always answer) | 100% | 62.5% | 100% | 60.9% |
| 0.5 | 63.9% | 73.9% | 65.1% | 70.7% |
| 0.6 | 52.8% | 73.7% | 46.7% | 81.0% |
| 0.7 | 44.4% | 75.0% | 36.8% | 83.0% |
| 0.8 | 25.0% | 94.4% | 23.0% | 92.9% |
| 0.9 | 12.5% | 100% | 14.5% | 90.9% |

Always read the accuracy together with the share of photos answered. Suggested app rule: below 0.7 confidence, show "not sure - take a clearer close-up of one leaf" and still show the top 3 guesses.

## 3. Per-class results

**PlantDoc test (72 images)**

| Class | Precision | Recall | F1 | Images |
|---|---|---|---|---|
| Pepper__bell___Bacterial_spot | 0.80 | 0.89 | 0.84 | 9 |
| Potato___Early_blight | 0.75 | 0.38 | 0.50 | 8 |
| Potato___Late_blight | 0.50 | 0.62 | 0.56 | 8 |
| Tomato_Bacterial_spot | 0.33 | 0.33 | 0.33 | 9 |
| Tomato_Early_blight | 0.50 | 0.78 | 0.61 | 9 |
| Tomato_Late_blight | 0.75 | 0.60 | 0.67 | 10 |
| Tomato_Septoria_leaf_spot | 0.70 | 0.64 | 0.67 | 11 |
| Tomato_healthy | 0.86 | 0.75 | 0.80 | 8 |
| **Macro average** | 0.65 | 0.62 | 0.62 | 72 |

**PlantWild test (304 images)**

| Class | Precision | Recall | F1 | Images |
|---|---|---|---|---|
| Pepper__bell___Bacterial_spot | 0.75 | 0.57 | 0.65 | 21 |
| Potato___Early_blight | 0.62 | 0.45 | 0.52 | 29 |
| Potato___Late_blight | 0.65 | 0.72 | 0.68 | 36 |
| Tomato_Bacterial_spot | 0.56 | 0.48 | 0.52 | 46 |
| Tomato_Early_blight | 0.50 | 0.63 | 0.56 | 52 |
| Tomato_Late_blight | 0.58 | 0.78 | 0.67 | 49 |
| Tomato_Septoria_leaf_spot | 0.55 | 0.37 | 0.44 | 30 |
| Tomato_healthy | 0.81 | 0.73 | 0.77 | 41 |
| **Macro average** | 0.63 | 0.59 | 0.60 | 304 |

**Main confusions (PlantDoc test):** tomato bacterial spot and Septoria are mixed up (3 and 4 photos); potato early and late blight are mixed up; some potato early blight photos are predicted as tomato early blight.

## 4. Model comparison (validation photos, used to choose the model)

| Model | PlantDoc val top-1 (147) | PlantWild val top-1 (146) | PlantWild val top-3 | Real-photo val macro-F1 | Training time | Parameters |
|---|---|---|---|---|---|---|
| **MobileNetV2 (selected)** | 55.8% | 69.9% | 90.4% | 0.623 | 6.9 min | 2.6 M |
| EfficientNetV2-S | 54.4% | 65.8% | 91.1% | 0.611 | 10.8 min | 20.7 M |

The two models are within the noise of these small validation sets. MobileNetV2 was selected because it is similar in accuracy and about 8 times smaller. Only the selected model was evaluated on the test sets.

## 5. Progress over the project (PlantDoc test, same 72 images)

| Setup | Top-1 |
|---|---|
| Original: PlantVillage only | 34.7% (25 of 72) |
| PlantDoc copied 5x, validation leaked (earlier notebook) | 48.6% (35 of 72) |
| Clean pipeline, PlantVillage + PlantDoc only (computed from its confusion matrix) | 44.4% (32 of 72) |
| **Final: PlantVillage + PlantWild + PlantDoc, MobileNetV2** | **62.5% (45 of 72)** |

The most likely reason for the improvement is the extra 1,228 real PlantWild training photos. The effect of each change was not tested separately.

## 6. Data used

| Set | Images |
|---|---|
| PlantVillage train / val | 9,748 / 1,647 |
| PlantWild train / val / test | 1,228 / 146 / 304 |
| PlantDoc train / val / test | 587 / 147 / 72 |

Near-duplicates removed: 3 (PlantDoc train vs test), 35 (PlantWild train vs PlantDoc val+test), 33 (PlantWild val vs training sets), 50 (PlantWild test vs training sets).

## 7. Training setup

- Input 288 x 288 RGB; each batch mixes PlantVillage 35%, PlantWild 50%, PlantDoc 15%
- Stage 1: train the new head with the backbone frozen (learning rate 1e-3); stage 2: fine-tune the top 40% of backbone layers (learning rate 5e-5), BatchNorm frozen
- Label smoothing 0.1, early stopping on real-photo validation accuracy, GPU mixed precision
- Augmentation: flips, rotation, zoom, translation, contrast, brightness

## 8. Using the model (for the API)

- Class order (output index 0-7): Pepper__bell___Bacterial_spot, Potato___Early_blight, Potato___Late_blight, Tomato_Bacterial_spot, Tomato_Early_blight, Tomato_Late_blight, Tomato_Septoria_leaf_spot, Tomato_healthy (saved in `class_names.json`)
- Input: RGB, 288 x 288, float32, raw 0-255 pixel values (do not divide by 255; the model has its own Rescaling layer)
- Resize: centre square crop of the shorter side, then bilinear resize
- Output: 8 softmax probabilities

## 9. Limitations

- The PlantDoc test set has only 72 images, so its accuracy is uncertain (95% interval roughly 51 - 74%). The PlantWild test (304 images) is a steadier estimate.
- PlantDoc and PlantWild are web-scraped: some labels are noisy, and some images are slides, collages or stock photos. Phone photos from real fields may behave differently.
- Only 8 classes are supported. Other crops or diseases get a confident-looking wrong answer unless the app uses the confidence threshold.
- PlantWild is licensed CC BY-NC-ND 4.0 (non-commercial). Do not share the dataset images; share only the model.
