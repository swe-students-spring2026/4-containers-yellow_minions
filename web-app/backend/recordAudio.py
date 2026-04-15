import pymongo
import sounddevice as sd
import soundfile as sf

"""
sample_rate: standard audio frequency rate
duration: length of recording
"""
sample_rate = 44100
duration = 5

def recordAudio():
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    return audio

def playAudio(recording):
    sd.play(recording, sample_rate)
    sd.wait()
