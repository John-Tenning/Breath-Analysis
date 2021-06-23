import math
import os
import matplotlib.pyplot  as plt
import librosa
import librosa.display
for root, folders, files in os.walk('F:\Studies\Sem 2\C Programming\Breath-Analysis\sound'):
    for folder in folders:
        os.makedirs(f'F:\Studies\Sem 2\C Programming\Breath-Analysis\img_data\{folder}',0o666)
        print(f'Processing {folder}')
        for _root, _folders, _files in os.walk(f'F:\Studies\Sem 2\C Programming\Breath-Analysis\sound\{folder}'):
            for file in _files:
                signal, sr = librosa.load(f'F:\Studies\Sem 2\C Programming\Breath-Analysis\sound\{folder}\{file}', sr=22050)
                size = signal.shape[0]
                start = 0
                end = start + math.floor(size/5)
                for i in range(5):
                    part_of_signal = signal[start:end]
                    mfcc = librosa.feature.mfcc(part_of_signal,
                                n_fft=2048,
                                hop_length=512,
                                n_mfcc=13)
                    librosa.display.specshow(mfcc, sr=sr, hop_length=512)
                    plt.savefig(f'F:\Studies\Sem 2\C Programming\Breath-Analysis\img_data\{folder}\{file[:-4] +"_"+ str(i)}.png')
                    plt.clf()
                    start = end
                    end = start + math.floor(size/5)
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator


training_set = ImageDataGenerator(rescale=1./255)
test_set = ImageDataGenerator(rescale=1./255)

import splitfolders
splitfolders.ratio('img_data', output="out", seed=1337, ratio=(.9, .1))

training_data = training_set.flow_from_directory(
        '/content/out/train',
        batch_size=1,
        target_size=(128, 128))

val_data = test_set.flow_from_directory(
        '/content/out/val',
        batch_size=1,
        target_size=(128, 128))

model = Sequential()
model.add(Conv2D(32, (3,3),activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(64, (3,3),activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(64, (3,3),activation='relu'))
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(5, activation='softmax'))
model.compile(optimizer=tf.keras.optimizers.Adam(3e-4),
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
