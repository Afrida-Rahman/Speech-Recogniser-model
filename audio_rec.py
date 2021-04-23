import sounddevice
from scipy.io.wavfile import write

fs=44100
second=3

record_voice=sounddevice.rec(int(second*fs),samplerate=fs,channels=2)
print('recording.......')
sounddevice.wait()
write("output.wav",fs,record_voice)
