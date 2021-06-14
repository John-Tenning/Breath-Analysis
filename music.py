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