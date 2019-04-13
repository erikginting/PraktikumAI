# In[01]: Soal No 2

kfold=StratifiedKFold(n_splits=5)
splits=kfold.split(d, d['CLASS'])

# In[02]: Soal No 9

model = Sequential()
model.add(Dense(512, input_shape=(2000 ,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

# In[03]: Soal No 10

model.compile(loss='categorical_crossentropy',optimizer='adamax', metrics=['accuracy'])

# In[04]: Soal No 3

train content = d['CONTENT'].iloc[train idx]
test content = d['CONTENT'].iloc[test idx]

# In[05]: Jawaban Teori No 3

import numpy as np
from sklearn.model_selection import train_test_split
X, y = np.arange(10).reshape((5, 2)), range(5)
X

# In[06]: Jawaban No 5

from keras.preprocessing.text import Tokenizer
ns = Tokenizer(num_words=2)
yn = ["Jawaban No5", "yes", "Berhasil"]
ns.fit_on_texts(yn)
ns.fit_on_sequences(yn)
ns.word_index

# In[07]: Soal No 6

d_train_inputs = tokenizer.texts to matrix(train content,mode='t
df') 
d_test_inputs = tokenizer.texts to matrix(test content, mode='t
df')

# In[08]: Jawaban No 7

d_train inputs = d_train_inputs/np.amax(np.absolute(d_train_inputs))
d_test_inputs = d_test_inputs/np.amax(np.absolute(d_test_inputs))

# In[09]: Jawaban No 8

d_train_outputs = np.utils.to_categorical(d['CLASS'].iloc[train_idx])
d_test_outputs = np.utils.to_categorical(d['CLASS'].iloc[test_idx]) 
