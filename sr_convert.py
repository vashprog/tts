import librosa
import os
import soundfile as sf

list_ = os.listdir("All_wavs")
for i,val in enumerate(list_):
  
    y, sr = librosa.load(f"All_wavs/{val}")
    data = librosa.resample(y, orig_sr =44100,target_sr= 22050)
    sf.write(f"wavs/{val}",data, samplerate =22050)