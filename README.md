# UTKFace Age Prediction

A small CNN that predicts a person's exact age from a face photo. The Streamlit app works with a webcam or an uploaded image.

**Live app:** https://utkface-age-prediction-0.streamlit.app/

## Files

- `UTKFace_Age_CNN_Final.ipynb` - data loading, training, evaluation
- `utkface_age_cnn.keras` - the trained model
- `app.py` - the Streamlit app
- `requirements.txt` - libraries for the app

## Data

[UTKFace](https://susanqq.github.io/UTKFace/): the age is in each file name (for example `26_0_2_...jpg`). Images are resized to 100x100 and split 80% train / 20% test, with 10% of the training data used for validation.

## Model

- 3 blocks of Conv2D (32, 64, 128 filters) + MaxPooling2D
- Dense layers (100, 64, 32, 16, 8, 4) with Dropout 0.2, then 1 output neuron for the age
- Regression: MSE loss, MAE metric, Adam optimizer
- Early stopping on validation loss (patience 3), up to 25 epochs, batch size 32

## Run the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Limitations

- Dark, bright or side-on photos give worse predictions.
- The model was trained on cropped faces, so keep your face centered and close in the webcam.
- UTKFace is not balanced across ages, ethnicities and genders, so the model may be less accurate for groups with fewer images.
- The prediction is an estimate that can be off by several years.
- Not for real decisions about people (age checks, hiring, access). It is a learning project.
