from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, Convolution2D, MaxPooling2D, Dropout
import numpy as np
import keras
print(keras.__version__)
import generate_training_set as gt

model = Sequential()
model.add(Convolution2D(19, 1, 4, input_shape=(1, 512, 19), border_mode='same',activation='tanh'))
# 4096 x 16
for nn in range(0,9):
    model.add(Convolution2D(19, 1, 2, border_mode='same', subsample=(1,2), activation='tanh'))
# 1 x 16
model.add(Convolution2D(1, 1, 1, border_mode='same', activation='relu'))
# 1
model.add(Flatten())
model.add(Dense(1))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('relu'))
model.add(Dense(256, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

for sentence_number in range(100,501):
    sentence_data = gt.generate_data(data_source_tag = 'arctic_a0', start_number = sentence_number, end_number = sentence_number+1, memory_length = 512)
    if len(sentence_data[0]) != 0:
        data = np.zeros((256))
        start_num = 0
        next_num = 1#len(sentence_data[0])
        end = 2#len(sentence_data[0])+1
        while(start_num + next_num < end):
            inputs = np.zeros((next_num,1,512,19), dtype='float32')
            print(sentence_number)
            targets = np.zeros((next_num,256), dtype='float32')
            for n in range(start_num,start_num+next_num):
                if n>1:
                    tmp_val = 512-(n+1-max(n-511,1))
                    tmp_zeros = np.zeros((tmp_val,19), dtype='float32')
                    actual_data = np.flipud(np.asarray(sentence_data[0][max(n-511,1):n+1], dtype='float32'))
                    inputs[n-start_num][0]= np.concatenate((actual_data,tmp_zeros), axis=0)
                    
                
                inputs[n-start_num][0][0] = np.asarray(sentence_data[0][n],dtype='float32')
                if (n > 0):
                    inputs[n-start_num][0][0][18] = sentence_data[0][n-1][18]
                else:
                    inputs[n-start_num][0][0][18] = 0
                
                
                targets[n-start_num][sentence_data[1][n][0]] = 1
            start_num += next_num
            model.fit(inputs, targets, nb_epoch=10)
            model.save('final_model')
            model.save_weights('final_weights')

