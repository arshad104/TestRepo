import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
import Pyro4

audio_server = Pyro4.Proxy("PYRONAME:audioserver")

all_wav_files = audio_server.convert_mp3_to_wav()

# key = all_wav_files[0]
# wav_file = audio_server.get_wav_files_by_keys(key)

# next_file = audio_server.iter_next()

# spf = wave.open(next_file["audio"],'rb')

# signal = spf.readframes(-1)
# signal = np.fromstring(signal, 'Int16')
# fs = spf.getframerate()

# if spf.getnchannels() == 2:
#     print 'Just mono files'
#     sys.exit(0)

# Time = np.linspace(0, len(signal)/fs, num=len(signal))

# plt.figure(1)
# plt.title('Audio Signal')
# plt.plot(Time,signal)
# plt.show()