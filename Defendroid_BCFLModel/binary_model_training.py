import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import math
import tensorflow as tf


def train_model(vulnerability_df):

    c_0 = vulnerability_df[vulnerability_df.Vulnerability_status == 0]
    c_1 = vulnerability_df[vulnerability_df.Vulnerability_status == 1]

    c_0_count = c_0.processed_code.count()
    c_1_count = c_1.processed_code.count()

    min_count = 0

    if c_0_count<=c_1_count:
        min_count = c_0_count
    else:
        min_count = c_1_count

    i = (math.ceil(min_count / 1000) * 1000)-1000
    print(min_count,i)

    df_0 = c_0.sample(i)
    df_1 = c_1.sample(i)

    vulnerability_df = pd.concat([df_0, df_1], ignore_index=True)

    vulnerability_df.Vulnerability_status.value_counts()

    code_list = vulnerability_df.processed_code.tolist()
    y = vulnerability_df.Vulnerability_status

    sentences = code_list
    y = y.values

    sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=0)
    vectorizer = CountVectorizer(analyzer = 'word', lowercase=True, max_df=0.80, min_df=10, ngram_range=(1,3),max_features=300)
    vectorizer.fit(sentences_train)
    X_train = vectorizer.transform(sentences_train).toarray()
    X_test  = vectorizer.transform(sentences_test).toarray()

    print(X_train.shape)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(20, input_shape=(X_train.shape[1],), activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.0001,
        patience=20,
        verbose=1,
        mode="auto",
        baseline=None,
        restore_best_weights=False,
    )

    return model,early_stopping,X_train,y_train,X_test,y_test