import pandas as pd
import numpy as np 
import os
DATA_DIR = "data"

#first i am making a function to extraction important features from the data given to us
# these includes rms(root mean square), mav(mean absolute value), and mean frequency {did use the fft from numpy}
def feature_extraction(data):
    window_size = 200
    sampling_len = 1000
    feature_list = []

    for start in range(0, len(data) - window_size, window_size):
        window = data.iloc[start: start + window_size]
        window_class = window['class'].mode()[0]
        row = {}
        for ch in ['channel1', 'channel2', 'channel3','channel4','channel5','channel6','channel7','channel8']:
            signal = window[ch]
            rms = np.sqrt(np.mean(signal ** 2))
            mav = np.mean(np.abs(signal))

            fft_vals = np.fft.fft(signal)
            fft_freq = np.fft.fftfreq(len(signal) , d= 1/sampling_len)
            fft_mag = np.abs(fft_vals)

            half = len(signal) // 2
            fft_freq_half = fft_freq[:half]
            fft_mag_half = fft_mag[:half]
            mean_frequency = np.sum(fft_freq_half  * fft_mag_half)/np.sum(fft_mag_half)

            row[f'{ch}_rms'] = rms
            row[f'{ch}_mav'] = mav
            row[f'{ch}_meanFreq'] = mean_frequency
        row['class'] = window_class
        feature_list.append(row)
    df = pd.DataFrame(feature_list)
    return df

#---------------------Joining all the 72 gesture.txt file into one single DataFrame----------------------
gesture_list = []
for i in range(1, 73):
    gesture_list.append(f'gesture{i}.txt')

gesture_all = []
for i, ges in enumerate(gesture_list, start= 1):
    gesture_df = pd.read_csv(os.path.join(DATA_DIR,ges), sep=r'\s+')
    feature_df = feature_extraction(gesture_df)
    feature_df['Subject_id'] = i
    gesture_all.append(feature_df)

fdf = pd.concat(gesture_all, ignore_index=True)


#-----------here i removed '0' and '7' from the column class because , zero meant nothing
# and 7 in the entirety of the dataframe( out 22k values) appeared only ~70 times , which would harm the 
#model's predicting ability , and '0' appeared more than 50% of the time , and actually meant nothing which would 
#again harm the models ability to predict the import class

final_df = fdf[(fdf['class'] != 0) & (fdf['class']!= 7)]




#-----------Training , fiting and testing the model------------------
train_range = list(range(1,58))
test_range = list(range(58, 73))
features = [col for col in final_df.columns if col  not  in ['class', 'Subject_id']]
train_df = final_df[final_df['Subject_id'].isin(train_range)]
test_df = final_df[final_df['Subject_id'].isin(test_range)]
train_X = train_df[features]
train_y = train_df['class']

test_X = test_df[features]
test_y = test_df['class']

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.dummy import DummyClassifier

dummymodel = DummyClassifier(strategy='most_frequent')
model = RandomForestClassifier(random_state=13)
model.fit(train_X, train_y)
dummymodel.fit(train_X, train_y)
dummy_predictions = dummymodel.predict(test_X)
predictions = model.predict(test_X)
accuracy_model = accuracy_score(test_y , predictions)
accuracy_dummy_model = accuracy_score(test_y , dummy_predictions)

#---------------printing the result---------------------
print(f'The base line accuracy of the trained model should be {accuracy_dummy_model * 100 :.2f}%')
print(f'The accuracy of the trained model is {accuracy_model * 100 :.2f}%')
print(classification_report(test_y, predictions))

