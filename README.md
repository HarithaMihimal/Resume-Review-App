# Resume Review and Prediction System

## Overview

This project consists of two main components:

1. **Streamlit Web Application**: A web app that takes a resume as input, processes it, and predicts the job category using a trained machine learning model.
2. **Jupyter Notebook**: A notebook for data exploration, cleaning, model training, and saving the trained models.

## Components

### 1. Streamlit Web Application

**File**: `app.py`

#### Requirements
- `streamlit`
- `pickle`
- `re`
- `nltk`
- `sklearn`

#### Features
- Upload a resume in `.txt` or `.pdf` format.
- The app processes the resume text and predicts the job category.
- Displays the predicted job category in a user-friendly format.

#### How to Run
1. Install the required packages using:
   ```bash
   pip install streamlit nltk scikit-learn
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

### 2. Jupyter Notebook

**File**: `resume_classification_notebook.ipynb`

#### Overview
- **Data Exploration**: Load and visualize resume dataset.
- **Data Cleaning**: Clean resume text data to remove unwanted characters and patterns.
- **Feature Extraction**: Use TF-IDF vectorizer to convert text data into numerical features.
- **Model Training**: Train a K-Nearest Neighbors (KNN) classifier with OneVsRest strategy.
- **Model Saving**: Save the trained model and TF-IDF vectorizer to disk using `pickle`.

#### Key Functions
- `clean_resume(resume_text)`: Cleans the resume text by removing URLs, hashtags, mentions, special characters, and extra spaces.
- `main()`: Runs the Streamlit app, processes uploaded resumes, and makes predictions.

#### Dependencies
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `sklearn`

#### How to Use
1. Load the dataset and clean the data.
2. Transform the cleaned data using TF-IDF vectorizer.
3. Train the KNN classifier and evaluate its performance.
4. Save the trained models to disk for later use in the Streamlit app.

## Example Usage

### Running the Streamlit App
1. Start the Streamlit server as described above.
2. Upload a resume file.
3. View the predicted job category.

### Using the Jupyter Notebook
1. Open the notebook in Jupyter.
2. Follow the steps to explore data, clean it, train the model, and save it.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements

- This project uses the NLTK library for text processing.
- The classification model was built using Scikit-learn.


![image](https://github.com/user-attachments/assets/0bc7206a-72fd-4907-9867-84416e8d0448)
