import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np

name = "metal"
song = "/Volumes/SMART FAT B/sins/music-cnn-classifier/model/Data/genres_original/"+name+"/"+name+".00001.wav"

signal, sr = librosa.load(song, sr=22050)
'''
librosa.display.waveplot(signal,sr=sr)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()
'''
#fast fourier transform
fft = np.fft.fft(signal)

mag = np.abs(fft)
freq = np.linspace(0,sr, len(mag))
freq = freq[:int(len(freq)/2)]
mag = mag[:int(len(mag)/2)]
'''
plt.plot(freq,mag)
plt.show()
'''
#stft -> spectogram
nFFT = 2048
hLen = 512

stft = librosa.core.stft(signal,hop_length = hLen, n_fft = nFFT)
spectrogram = np.abs(stft)
spectrogram = librosa.amplitude_to_db(spectrogram)
'''
librosa.display.specshow(spectrogram, sr= sr, hop_length = hLen)
plt.xlabel("Time")
plt.ylabel("Freq")
plt.title("db")
plt.colorbar()
plt.show()
'''

#MFCCs
mfcc = librosa.feature.mfcc(signal,n_fft=nFFT,hop_length= hLen, n_mfcc=10)
librosa.display.specshow(mfcc, sr=sr, hop_length = hLen)
plt.xlabel("Time")
plt.ylabel("MFCC")
plt.colorbar()
plt.show()



