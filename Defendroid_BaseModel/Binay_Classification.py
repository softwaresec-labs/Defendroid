import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,f1_score,accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import math
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf

vulnerability_df = pd.read_pickle("proccessed_dataset_for_analysis.pickle")

vulnerability_df.Vulnerability_status.value_counts()

c_0 = vulnerability_df[vulnerability_df.Vulnerability_status == 0]
c_1 = vulnerability_df[vulnerability_df.Vulnerability_status == 1]

c_0_count = c_0.processed_code.count()
c_1_count = c_1.processed_code.count()

min_count = 0

if(c_0_count<=c_1_count):
    min_count = c_0_count
else:
    min_count = c_1_count

i = (math.ceil(min_count / 1000) * 1000)-1000
print(min_count,i)

df_0 = c_0.sample(i)
df_1 = c_1.sample(i)

vulnerability_df = pd.concat([df_0, df_1], ignore_index=True)

print(vulnerability_df.Vulnerability_status.value_counts())

code_list = vulnerability_df.processed_code.tolist()
y = vulnerability_df.Vulnerability_status

sentences = code_list
y = y.values

sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.20, random_state=1)
vectorizer = CountVectorizer(analyzer = 'word', lowercase=True, max_df=0.80, min_df=40, ngram_range=(1,3))
vectorizer.fit(sentences_train)
X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)

print(len(vectorizer.vocabulary_))

print(X_train.shape,y_train.shape)

model = Sequential()

model.add(Dense(units=20, activation ='relu')) # Input Layer
model.add(Dense(1, activation = 'sigmoid')) # Output Layer

model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics = ['accuracy'])

# Early Stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0.0001,
    patience=20,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
)

model_history = model.fit(x = X_train, y=y_train, epochs =1000, callbacks=early_stopping, validation_data=(X_test,y_test))

print(model.summary())

print(model_history.history.keys())

import matplotlib.pyplot as plt
plt.plot(model_history.history['accuracy'], label='accuracy')
plt.plot(model_history.history['val_accuracy'], label='val_accuracy')
plt.title("Model Accuracy")
plt.ylabel('accuracy')
plt.ylabel('val_accuracy')
plt.xlabel('epoch')
plt.legend(loc='lower right')
plt.show()

plt.plot(model_history.history['loss'], label='loss')
plt.plot(model_history.history['val_loss'], label='val_loss')
plt.title("Model Loss")
plt.ylabel('loss')
plt.ylabel('val_loss')
plt.xlabel('epoch')
plt.legend(loc='lower right')
plt.show()

print(y_train)

prediction = model.predict(X_test)

prediction = (model.predict(X_test) > 0.5).astype("int32")
print(prediction)

print(prediction[:5])

print(y_test[:5])

my_accuracy = accuracy_score(y_test,prediction.round())
print(my_accuracy)

my_f1_score = f1_score(y_test,prediction.round())
print(my_f1_score)

import seaborn as sn
cm = confusion_matrix(y_test, prediction.round())
print(cm)
print(classification_report(y_test, prediction.round()))
sn.heatmap(cm, annot=True, fmt='g')

model.save('binary_model.keras')

with open("binary_model.pickle", 'wb') as fout:
    pickle.dump((vectorizer, model), fout)