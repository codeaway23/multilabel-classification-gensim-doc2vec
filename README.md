# Multilabel Classification with Gensim Doc2Vec

---
## Dataset information

The dataset consists of customer reviews and multiple labels for different reviews. Total number of classes is 15. Total number of training reviews are 41569 and testing reviews are 10393. 

**Label counts**  

![label counts]()

**Number of reviews with multiple labels**  

![multiple labels]()

## Methodology

### Preprocessing
- All reviews turned to lowercase
- All stopwords removed
- All special characters removed

### Training
- Doc2Vec model followed by a logistic regression model is used
```python
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
```