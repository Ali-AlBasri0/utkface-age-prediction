# UTKFace Age Prediction

A small CNN that predicts a person's age from a face photo, with a Streamlit app that works with a webcam or an uploaded image.

**Live app:** https://utkface-age-prediction-0.streamlit.app/

## What it does

- Takes a face image (webcam or upload)
- Predicts the age in years
- Shows the age group: Child (0-12), Teen (13-19), Young adult (20-35), Adult (36-55), Senior (56+)

## Files

- `UTKFace_Age_CNN_Final.ipynb` - data loading, exploration, training, evaluation
- `utkface_age_cnn.keras` - the trained model
- `app.py` - the Streamlit app
- `requirements.txt` - libraries for the app

## Dataset

[UTKFace](https://susanqq.github.io/UTKFace/): face images where the age is in the file name (for example `26_0_2_...jpg`). Images are resized to 100x100 and split 80% train / 20% test. 10% of the training data is used for validation.

## Model

- Input 100x100x3, pixels scaled to 0-1 with a Rescaling layer
- 3 blocks of Conv2D (32, 64, 128 filters, 3x3, ReLU) + MaxPooling2D (2x2)
- Flatten, then Dense layers of 100, 64, 32, 16, 8, 4 units (ReLU) with Dropout 0.2 between some of them
- Output: 1 neuron (the age)
- Loss: mean squared error, metric: MAE, optimizer: Adam
- Early stopping on validation loss (patience 3, best weights restored), up to 25 epochs, batch size 32

This is a regression model (it predicts the age as a number). The age group in the app is made from the predicted age.

## Run the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501.

## Limitations and ethics

- **Lighting and angle:** dark, bright or side-on photos give worse predictions.
- **Background:** the model was trained on cropped faces. A webcam photo with a lot of background around the face can hurt accuracy, so keep your face centered and close.
- **Data bias:** UTKFace is not balanced across ages, ethnicities and genders, so the model may be less accurate for groups that have fewer images.
- **Not exact:** the prediction is an estimate with an error of several years. Glasses, makeup, beards and filters can also change the result.
- **Do not use it for decisions** about people (age checks, hiring, access, etc.). It is a learning project.
