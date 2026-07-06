# EMG Gesture Classification

Classifies hand gestures from 8-channel EMG (electromyography) signals using engineered time/frequency-domain features and classical ML models.

## Overview
- **Input**: 8-channel EMG signal recordings from 36 subjects(2 sets from each subject), each labeled with a gesture class (1–6; class 0 = unmarked, class 1-6 = hand/wrist movement  class 7 = near-absent and excluded).

- **Features extracted per 200-sample window per channel**: RMS, MAV (mean absolute value), and mean frequency (via FFT).
- **Models**: Random Forest and KNN (with StandardScaler), benchmarked against a DummyClassifier baseline(strategy = 'most_frequent').
- **Result**: ~89% accuracy with Random Forest (vs. 16.6% baseline).

## Dataset
[https://archive.ics.uci.edu/static/public/481/emg+data+for+gestures.zip]. Download all the data from the link  and place them in `data/`.

## Setup
1. `pip install -r requirements.txt`
2. Download and unzip the dataset from [UCI link] into `EMG_data_for_gestures-master/`
3. Run `bash prepare_data.sh` — this collects all .txt files from every subject folder into `data/gesture1.txt` ... `gesture72.txt`
4. Run `python main.py`
## Usage
```bash
python main.py
```
or open `EMG_gesture.ipynb` for the exploratory/step-by-step version.

## Approach
1. Concatenate all 72 subject files, tagging each with a `Subject_id`.
2. Window each channel's signal (window size = 200 samples) and extract RMS, MAV, mean frequency per window.created a function feature_extraction
3. Drop class 0 (unmarked) and class 7 (only ~70 occurrences out of ~22k rows) —i dropped them because on inspection they dropped performance of the model
4. Split by subject (57 train / 15 test) rather than randomly, to avoid leaking a subject's signal patterns across train/test , I didn't use the regular train_test_split because this is a time-series data and doing it with a basic train_test_split would result in a data leakage
5. Train Random Forest and KNN, compare against a majority-class dummy baseline.(i used both randomforestclassifier and KNN because i thought it would help me comapre both the model)

## Results
## Results
| Model | Accuracy |
|---|---|
| Dummy (most frequent) | 16.66% |
| Random Forest (single train/test split) | 85.83% |
| Random Forest (5-fold cross-validation, grouped by subject) | 87.59% ± 1.73% |
| KNN (k=7) | 84.43% |

accuracy varies by 1-2% depending on which subjects land in train vs. test — 
this is expected variance, not a bug. Cross-validation gives a more stable estimate 
than a single split.
In the notebook I plotted confusion matrix to see how each class is confused by the model.
when the class-0 was put in most of the predicted class were predicted to be '0' , obv because class-0 had the highest frequency of any class, and it was an unmarked data so it was dropped
dropping it resulted in an increase of the model's accuracy

Feature importance shows RMS and MAV dominate; mean frequency contributes far less.

NOTE:
1) This is my first project where I used windowing and feature extraction , the data was very large and I learnt and implemented important methods to skim the important parts from the data

2) I have not been able to implement CNN , but I'll soon put an update where I put CNN to these models

3) Since the split is by file rather than by unique person, there's a small chance both sessions of the same subject aren't fully isolated between train/test. A cleaner version would group by actual subject ID (available in the original dataset metadata) rather than by file order.

4) The notebook version is the exploratory original; main.py is the cleaned-up version
