from model import *

import pandas as pd
from matplotlib import pyplot as plt

no_epochs = 60
inference_steps = 40
val_split_ratio = 0.2

## file paths
data_file = 'data/train_data.csv'
label_file = 'data/train_label.csv'
data_df = pd.read_csv(data_file)
label_df = pd.read_csv(label_file)

##visualising training data
label_df['label'].value_counts().plot(kind="bar")
plt.title('Labels v/s Label Counts')
plt.show()

rem_duplicates = label_df['label'].groupby(label_df.id).apply(set).reset_index()
pd.Series([len(x) for x in rem_duplicates['label'].values]).value_counts().plot('bar')
plt.title('No of IDs with multiple labels')
plt.xlabel('No of labels per ID')
plt.ylabel('No of IDs')
plt.show()

## preparing training data
text_df = prepare_training_data(data_file, label_file)

## building the document vector model
model_dbow, logreg = Doc2VecModel(text_df, no_epochs, inference_steps, val_split_ratio)
print("Pickling the models...")
with open('dbow_model.pkl','wb') as file:
	pickle.dump(model_dbow, file)
with open('log_reg_model.pkl','wb') as file:
	pickle.dump(logreg, file)

## preparing submission
test_file = 'data/test_data.csv'
submission_df = submission(model_dbow, logreg, test_file, inference_steps)
