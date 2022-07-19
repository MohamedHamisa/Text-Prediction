import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from keras.utils.data_utils import get_file
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Input, Embedding, Reshape, merge, LSTM, Bidirectional
from keras.layers import SimpleRNN, TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv1D, MaxPooling1D, ZeroPadding1D
from keras.utils import np_utils
from keras.optimizers import Adam
import cPickle as pickle
import bcolz #يتم تخزين البيانات في شكل عمدان مضغوطة و ممكن اشيل و احط داتا بسهولة و مابياخدش مكان كبير في الميموري
import re
from numpy.random import random, permutation, randn, normal, uniform, choice

path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
text = open(path).read()
print len(text)

chars = sorted(list(set(text)))  #هحط التيكست في ليست علشان اشيل المتكرر وبعدين يحطها في ليست و يصنفها 
print len(chars)+1 #عمل بلس وان علشان يحط الزيرو اللي مش موجود

chars.insert(0, '\0')

char_to_index = {v:i for i,v in enumerate(chars)} 
index_to_char = {i:v for i,v in enumerate(chars)}

total_index = [char_to_index[char] for char in text]  #هيمسك كل حرف و يحوله للرقم المنسب ليه من الديكشنري اللي فوق

total_index[:10]

''.join(index_to_char[i] for i in total_index[:25])


#هنا عمل الاكس مجموعة من ليستات مكونة من 7 حروف و الواي هو الحرف التامن
pred_num = 25
xin = [[total_index[j+i] for j in xrange(0, len(total_index)-1-pred_num, pred_num)] for i in range(pred_num)]
y = [total_index[i+pred_num] for i in xrange(0, len(total_index)-1-pred_num, pred_num)]

X = [np.stack(xin[i][:-2]) for i in range(pred_num)]
Y = np.stack(y[:-2])

Y[:8]
X[0].shape, Y.shape

hidden_layers = 256
vocab_size = 86
n_fac = 42

model = Sequential([
        Embedding(vocab_size, n_fac, input_length=pred_num),
        SimpleRNN(hidden_layers, activation='relu'),
        Dense(vocab_size, activation='softmax')
    ])

model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam())

model.fit(np.stack(X, 1), Y, batch_size=64, epochs=5)

model.save_weights('simpleRNN_3pred.h5')

model.load_weights('simpleRNN_3pred.h5')

model.save_weights('simpleRNN_7pred.h5')

model.load_weights('simpleRNN_7pred.h5')

def predict_next_char(inp):
    index = [char_to_index[i] for i in inp]
    arr = np.expand_dims(np.array(index), axis=0)
    prediction = model.predict(arr)
    return index_to_char[np.argmax(prediction)]

predict_next_char('those w')

predict_next_char(' th')

predict_next_char(' an')


predict_next_char('does th')

ys = [[total_index[j+i] for j in xrange(1, len(total_index)-pred_num, pred_num)] for i in range(pred_num)]

Y_return = [np.stack(ys[i][:-2]) for i in range(pred_num)]

vocab_size = 86
n_fac = 42
hidden_layers = 256

return_model = Sequential([
        Embedding(vocab_size, n_fac, input_length=pred_num),
        SimpleRNN(hidden_layers, return_sequences=True, activation='relu'),
        TimeDistributed(Dense(vocab_size, activation='softmax'))
    ])

return_model.summary()

return_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam())

X_model = np.stack(X, 1)
Y_model = np.expand_dims(np.stack(Y_return, 1), axis=-1)

return_model.fit(X_model, Y_model, batch_size=64, epochs=5)

return_model.optimizer.lr = 1e-4
return_model.fit(X_model, Y_model, batch_size=64, epochs=5)

return_model.optimizer.lr = 1e-4
return_model.fit(X_model, Y_model, batch_size=64, epochs=5)

return_model.save_weights('return_sequences_25.h5')

#فانكشن بيجرب بقي
def predict_every_char(inp):
    l = []
    p = 0
    while p<len(inp):
        pre_inp = inp[p:p+pred_num]
        if len(pre_inp) < pred_num:
            pre_inp = pre_inp + ' '*(pred_num - len(pre_inp))
            l.append(pre_inp)
        else:
            l.append(pre_inp) 
        p+=pred_num

#     index = [char_to_index[i] for i in inp]
#     arr = np.expand_dims(index, axis=0)
#     prediction = return_model.predict(arr)
#     return ''.join([index_to_char[np.argmax(i)] for i in prediction[0]])
    
    final = []
    for half in l:
        index = [char_to_index[i] for i in half]
        arr = np.expand_dims(index, axis=0)
        prediction = return_model.predict(arr)
        final.append(''.join([index_to_char[np.argmax(i)] for i in prediction[0]]))
    
    return ''.join(final)

predict_every_char('and the boy left')

predict_every_char('this is')

predict_every_char("140 After having discovered in many of the less comprehensible actions mere manifestations of pleasure in emotion for its own sake, I fancy I can detect in the self contempt which characterises holy persons, and also in their acts of self torture (through hunger and scourgings, distortions and chaining of the limbs, acts of madness) simply a means whereby such natures may resist the general exhaustion of their will to live (their nerves). They employ the most painful expedients to escape if only for a time from the heaviness and weariness in which they are steeped by their great mental indolence and their subjection to a will other than their own.")

bs = 64

stateful_model = Sequential([
        Embedding(vocab_size, n_fac, input_length=pred_num, batch_input_shape=(bs, 7)),
        BatchNormalization(),
        LSTM(hidden_layers, activation='tanh', return_sequences=True, stateful=True),
        TimeDistributed(Dense(vocab_size, activation='softmax'))
    ])

stateful_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam())

divide = len(X_model)//bs*bs

stateful_model.fit(X_model[:divide], Y_model[:divide], batch_size=64, epochs=5, shuffle=False)

stateful_model.optimizer.lr = 1e-4
stateful_model.fit(X_model[:divide], Y_model[:divide], batch_size=64, epochs=5, shuffle=False)

# def predict_every_char_stateful(inp):
#     index = [char_to_index[i] for i in inp]
#     arr = np.expand_dims(index, axis=0)
#     arr = np.resize(arr, (bs, 7))
#     prediction = stateful_model.predict(arr, batch_size=64)
#     return [index_to_char[np.argmax(i)] for i in prediction[0]]  


def predict_every_char_stateful(inp):
    l = []
    p = 0
    while p<len(inp):
        pre_inp = inp[p:p+pred_num]
        if len(pre_inp) < pred_num:
            pre_inp = pre_inp + ' '*(pred_num - len(pre_inp))
            l.append(pre_inp)
        else:
            l.append(pre_inp) 
        p+=pred_num
    
    final = []
    for half in l:
        index = [char_to_index[i] for i in half]
        arr = np.expand_dims(index, axis=0)
        arr = np.resize(arr, (bs, 7))
        prediction = stateful_model.predict(arr, batch_size=64)
        final.append(''.join([index_to_char[np.argmax(i)] for i in prediction[0]]))
    return ''.join(final)

 predict_every_char_stateful('this is')

predict_every_char_stateful("140 After having discovered in many of the less comprehensible actions mere manifestations of pleasure in emotion for its own sake, I fancy I can detect in the self contempt which characterises holy persons, and also in their acts of self torture (through hunger and scourgings, distortions and chaining of the limbs, acts of madness) simply a means whereby such natures may resist the general exhaustion of their will to live (their nerves). They employ the most painful expedients to escape if only for a time from the heaviness and weariness in which they are steeped by their great mental indolence and their subjection to a will other than their own.")

