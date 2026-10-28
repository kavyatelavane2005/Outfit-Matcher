# Concepts behind Outfit Matcher

This project combines **deep learning** (a classifier and an image generator) with **classical computer vision**
and **simple rules**. This page explains each idea in plain language.

## 1. The problem
Given one garment, suggest what to wear with it. An outfit here means **shirt + pants + shoes**. If you upload one of
them, the system produces the other two. A dress covers both shirt and pants, so only shoes are generated.

## 2. Convolutional neural network (CNN) and ResNet18
A CNN learns to recognise images by stacking layers of small filters. Early layers detect edges and textures, later
layers detect shapes and whole objects. **ResNet18** has 18 layers and uses *residual (skip) connections*, which let
signals bypass layers so deep networks train reliably.

## 3. Transfer learning
Training a CNN from scratch needs millions of images. ResNet18 was already trained on **ImageNet** (1.2 million
photos), so it already knows edges, fabrics and shapes. We keep that knowledge and only replace the **last layer**
with one that outputs our 15 garment types. This needs far less data (about 9,000 photos) and time (about 3 minutes).

## 4. Two-phase fine-tuning
1. **Head only:** freeze the whole backbone, train only the new last layer (learning rate 1e-3, 3 epochs).
2. **Fine-tune:** also unfreeze the last convolutional block (`layer4`) and train gently (learning rate 1e-4, 4 epochs).

A small learning rate in phase 2 avoids destroying the useful pretrained features. Batch-norm statistics stay frozen.
After training, the weights of the **best epoch by validation accuracy** are restored, not simply the last epoch.

## 5. Data augmentation
Each training photo is randomly cropped, shifted, zoomed out, rotated slightly, blurred, flipped and colour-jittered.
The model sees many variants, so it copes better with photos that look different from the catalogue.

## 6. Class-weighted loss
The dataset has many more T-shirts than, say, skirts. The cross-entropy loss gives rare classes a higher weight so the
model does not simply favour the common ones. Training data is also capped at 800 photos per class.

## 7. Train, validation and test sets
Data is split 70% / 15% / 15%. The **validation** set chooses the best epoch. The **test** set is used once at the end,
so its score is an honest estimate. The planner needs the *role* (top, bottom, dress, shoes), so both 15-type accuracy
and 4-role accuracy are reported. Confusing a shirt with a T-shirt does not matter for planning; confusing a shirt with
shoes does.

## 8. Confidence and "not sure"
The classifier outputs probabilities (softmax). If the best one is below **55%**, the system says it is not sure and
asks the user to pick the garment type, instead of planning an outfit around a wrong guess.

## 9. Finding the garment: background removal (classical)
A photo contains hangers, walls and shadows that would confuse the next steps.
- **Background model:** the border pixels are assumed to be background. K-means learns their colour range (including
  shadows and gradients). Anything far from that range is garment. The largest connected blob is kept.
- **GrabCut:** if the border is busy, OpenCV's GrabCut is used, seeded with "the garment is around the middle".
- **Fallback:** the centre of the picture.

## 10. Reading the colour: k-means and CIE Lab
**K-means** groups the garment's pixels into 3 clusters; the biggest cluster is the garment's main colour. The result
is matched to the nearest of 25 named fashion colours (Black, Navy Blue, Olive, Mustard...). Distances are measured in
**CIE Lab** space, where numeric distance (delta-E) matches how different colours look to people. RGB distance does not.

## 11. Colour-theory rules for matching
No model is trained on real outfits. Instead:
- **Neutral garments** (white, black, grey, navy, beige...) use hand-written stylist pairings.
- **Coloured garments** use harmony on the colour wheel: Look 1 = safe neutral, Look 2 = **analogous** (neighbouring
  hue, muted and darker), Look 3 = **complementary** (opposite hue, muted). Loud colours are excluded as partners.
- **Shoes** are chosen to suit the trousers or skirt (white, black or brown depending on how earthy or dark they are).

## 12. Diffusion model and SD-Turbo
A diffusion model generates an image by starting from random noise and removing noise step by step, steered by a text
prompt such as *"studio product photo of men's navy blue jeans on a plain grey background"*. It works in a compressed
**latent space** and a VAE decoder turns the result into pixels. **SD-Turbo** is a distilled version that needs only
**2 steps**, so each image takes about a second on a GPU. It is pretrained; we do not train it.

## 13. Checking generated images
Generators do not always obey the colour in the prompt. Each generated image is tested twice:
1. **Colour check:** its dominant colour must be within delta-E 25 of the planned colour.
2. **Type check:** our own classifier must agree it is the right role (e.g. "bottom").
Up to 4 tries (different seeds) are made and the best is kept.

## Deep learning vs classical parts
| Deep learning | Classical / rule-based |
|---|---|
| ResNet18 garment classifier (fine-tuned) | Background removal (k-means, GrabCut) |
| SD-Turbo image generator (pretrained) | Colour reading (k-means), Lab distance |
| | Colour-harmony outfit planner |

## Glossary
- **Epoch:** one pass over the training data.
- **Softmax:** turns model scores into probabilities that sum to 1.
- **Overfitting:** memorising training photos instead of learning general features; validation data detects it.
- **Delta-E:** numeric distance between two colours in Lab space.
