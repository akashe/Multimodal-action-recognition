from scipy.io import wavfile
import os



max_len = 0
total_files = 0
avg_len = 0
data_loc = "../data/wavs/speakers/"
for root,dirs,files in os.walk(data_loc):
    for file in files:
        sample_rate, data = wavfile.read(os.path.join(root,file))
        if len(data)>max_len:
            max_len= len(data)
        avg_len += len(data)
        avg_len /= 2
        total_files += 1

print(max_len)
print(total_files)
print(avg_len)


wav_location = "../data/wavs/speakers/8B9N9jOOXGUordVG/0b5b2220-45e5-11e9-b578-494a5b19ab8b.wav"

sample_rate, data = wavfile.read(wav_location)
print("Akash")