import pickle
import gensim
import numpy as np
import pandas as pd
import multiprocessing
from tqdm import tqdm
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn import utils
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

import text_preprocessing

def prepare_training_data(data_file, label_file):
	## read training data as a pandas dataframe
	orig_df = pd.read_csv(data_file)
	## text preprocessing
	text = pd.Series.tolist((orig_df['text']))
	x_id = pd.Series.tolist((orig_df['id']))
	text = text_preprocessing.normalize_sent_list(text,
                        		lowercase=True,
                        		stopwords=True,
                        		specialchar=True,
                        		lemmatize=False)
	## preparing preprocessed text
	text_df = pd.DataFrame(text, columns=["text"])
	text_df["id"] = x_id
	text_df = text_df.set_index('id')
	## reading label data as a pandas dataframe
	label_df = pd.read_csv(label_file)
	## preparing label data
	label_df = label_df["label"].groupby(label_df.id).apply(list).reset_index()
	label_df = label_df.set_index('id')
	## concat text data and label data
	text_df = pd.concat([text_df,label_df], axis=1)
	return text_df

def Doc2VecModel(text_df, no_epochs, inference_steps, val_split_ratio):
	## splitting dataframe into training and validation frames
	train_df, val_df = train_test_split(text_df, test_size=val_split_ratio)	
	## creating tagged documents
	train_tagged = train_df.apply(
		lambda r: TaggedDocument(words=r['text'].split(), tags=r.label), axis=1)
	val_tagged = val_df.apply(
		lambda r: TaggedDocument(words=r['text'].split(), tags=r.label), axis=1)
	## building a distributed bag of words model 
	cores = multiprocessing.cpu_count()
	print("Building the Doc2Vec model vocab...")
	model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, min_count=2, workers=cores)
	model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])
	## training the model
	print("Training the Doc2Vec model for ", no_epochs, "number of epochs" )
	for epoch in range(no_epochs):
		model_dbow.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), 
				total_examples=len(train_tagged.values), epochs=1)
		model_dbow.alpha -= 0.002
		model_dbow.min_alpha = model_dbow.alpha
	## preparing document vectors for learning
	def vec_for_learning(model, tagged_docs):
		sents = tagged_docs.values
		targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=inference_steps)) for doc in sents])
		return targets, regressors
	y_train, X_train = vec_for_learning(model_dbow, train_tagged)
	y_val, X_val = vec_for_learning(model_dbow, val_tagged)
	## training a logistic regression model
	print("Training the logistic regression model...")
	logreg = LogisticRegression(solver='lbfgs', multi_class='auto')
	logreg.fit(X_train, y_train)
	## making predictions on the training set
	print("Prediction numbers:")
	train_binary = logreg.predict(X_train)
	print('Accuracy on the training set : %s' % accuracy_score(y_train, train_binary))
	print('F1 score on the training set : {}'.format(f1_score(y_train, train_binary, average='weighted')))
	## making predictions on the validation set
	val_binary = logreg.predict(X_val)
	print('Accuracy on the validation set : %s' % accuracy_score(y_val, val_binary))
	print('F1 score on the validation set : {}'.format(f1_score(y_val, val_binary, average='weighted')))
	return model_dbow, logreg

def submission(model_dbow, logreg_model, test_file, inference_steps):
	test_df = pd.read_csv(test_file)
	## preparing document vectors for testing
	def vec_for_testing(model_dbow, test_df, inference_steps):
		test_text = test_df['text'].values
		for n,text in enumerate(test_text):
			test_text[n] = text.split()
		regressors = tuple([model_dbow.infer_vector(text,steps=inference_steps) for text in test_text])
		return regressors
	print("Preparing test vectors...")
	X_test = vec_for_testing(model_dbow, test_df, inference_steps)
	## using model to generate class probabilities
	print("Recording probabilities...")
	y_pred = logreg_model.predict_proba(X_test)
	classes = logreg_model.classes_
	submission_df = pd.DataFrame(y_pred, index=test_df.id, columns=classes)
	print("Saving submission data as a csv file...")	
	submission_df.to_csv('submission.csv')
	return submission_df
