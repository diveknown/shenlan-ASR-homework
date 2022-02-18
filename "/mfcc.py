#!/usr/bin/env python
# coding: utf-8

# In[1]:


import librosa
import numpy as np
from scipy.fftpack import dct


# In[29]:


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# In[30]:


def plot_spectrogram(spec, note,file_name):
#     """Draw the spectrogram picture
#         :param spec: a feature_dim by num_frames array(real)
#         :param note: title of the picture
#         :param file_name: name of the file
#     """ 
    fig = plt.figure(figsize=(20, 5))
    heatmap = plt.pcolor(spec)
    fig.colorbar(mappable=heatmap)
    plt.xlabel('Time(s)')
    plt.ylabel(note)
    plt.tight_layout()
    plt.savefig(file_name)


# In[31]:


#preemphasis config 
alpha = 0.97


# In[32]:


# Enframe config
frame_len = 400      # 25ms, fs=16kHz
frame_shift = 160    # 10ms, fs=15kHz
fft_len = 512


# In[33]:


# Mel filter config
num_filter = 23
num_mfcc = 12


# In[36]:


# Read wav file
wav, fs = librosa.load('C:/Users/guini/Desktop/ASR_Course-master/ASR_Course-master/02-feature-extraction/test.wav', sr=None)
wav.size


# In[37]:


# Enframe with Hamming window function
def preemphasis(signal, coeff=alpha):
    """perform preemphasis on the input signal.

        :param signal: The signal to filter.
        :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.97.
        :returns: the filtered signal.
    """
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


# In[38]:


def enframe(signal, frame_len=frame_len, frame_shift=frame_shift, win=np.hamming(frame_len)):
    """Enframe with Hamming widow function.

        :param signal: The signal be enframed
        :param win: window function, default Hamming
        :returns: the enframed signal, num_frames by frame_len array
    """
    
    num_samples = signal.size
    num_frames = np.floor((num_samples - frame_len) / frame_shift)+1
    frames = np.zeros((int(num_frames),frame_len))
    for i in range(int(num_frames)):
        frames[i,:] = signal[i*frame_shift:i*frame_shift + frame_len] 
        frames[i,:] = frames[i,:] * win                 #hanming窗函数

    return frames


# In[20]:


def get_spectrum(frames, fft_len=fft_len):                            #fft
    """Get spectrum using fft
        :param frames: the enframed signal, num_frames by frame_len array
        :param fft_len: FFT length, default 512
        :returns: spectrum, a num_frames by fft_len/2+1 array (real)
    """
    cFFT = np.fft.fft(frames, n=fft_len)
    valid_len = int(fft_len / 2 ) + 1
    spectrum = np.abs(cFFT[:,0:valid_len])
    return spectrum


# In[117]:


def fbank(spectrum, num_filter = num_filter):
    """Get mel filter bank feature from spectrum
        :param spectrum: a num_frames by fft_len/2+1 array(real)
        :param num_filter: mel filters number, default 23
        :returns: fbank feature, a num_frames by num_filter array 
        DON'T FORGET LOG OPRETION AFTER MEL FILTER!
    """
    
    pow_frames = ((1.0 / 512) * ((spectrum) ** 2))                                                      #功率谱
    feats=np.zeros((spectrum.shape[0], num_filter))
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (fs / 2) / 700))                         # 将Hz转换为Mel，采样率16000，最高也就8Khz
    mel_points = np.linspace(low_freq_mel, high_freq_mel, num_filter + 2)            # 由设计的滤波器数量得到的Mel频率
    hz_points = (700 * (10**(mel_points / 2595) - 1))                              #对应的线性频率域上的频率
    bin = np.floor((512 + 1) * hz_points / fs)                                    #频率对应在帧上的采样点
    fbank = np.zeros((num_filter, int(np.floor(512 / 2 + 1))))                    #每个滤波器对应每一个频率采样点上的功率
    #print(fbank.shape)

    for m in range(1, num_filter + 1):
        f_m_minus = int(bin[m - 1])   # 左
        f_m = int(bin[m])             # 中
        f_m_plus = int(bin[m + 1])    # 右

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    feats = np.dot(pow_frames, fbank.T)
    feats = np.where(feats == 0, np.finfo(float).eps, feats)  # 数值稳定性
    feats = 20 * np.log10(feats)  # dB

    return feats


# In[120]:


def mfcc(fbank, num_mfcc = num_mfcc):
    """Get mfcc feature from fbank feature
        :param fbank: a num_frames by  num_filter array(real)
        :param num_mfcc: mfcc number, default 12
        :returns: mfcc feature, a num_frames by num_mfcc array 
    """

    feats = dct(fbank, type=2, axis=1, norm='ortho')[:, 1 : (num_mfcc + 1)]
    return feats


# In[124]:


def write_file(feats, file_name):
    """Write the feature to file
        :param feats: a num_frames by feature_dim array(real)
        :param file_name: name of the file
    """
    f=open(file_name,'w')
    (row,col) = feats.shape
    for i in range(row):
        f.write('[')
        for j in range(col):
            f.write(str(feats[i,j])+' ')
        f.write(']\n')
    f.close()


# In[126]:


def main():
    wav, fs = librosa.load('C:/Users/guini/Desktop/ASR_Course-master/ASR_Course-master/02-feature-extraction/test.wav', sr=None)
    print(fs)
    signal = preemphasis(wav)
    #print(signal.shape)
    frames = enframe(signal)
    #print(frames.shape)
    spectrum = get_spectrum(frames)
    #print(spectrum.shape)
    fbank_feats = fbank(spectrum)
    #print(fbank_feats.shape)
    mfcc_feats = mfcc(fbank_feats)
    #print(mfcc_feats.shape)
    plot_spectrogram(fbank_feats.T, 'Filter Bank','fbank.png')
    write_file(fbank_feats,'./test.fbank')
    plot_spectrogram(mfcc_feats.T, 'MFCC','mfcc.png')
    write_file(mfcc_feats,'./test.mfcc')

if __name__ == '__main__':
    main()


# In[ ]:




