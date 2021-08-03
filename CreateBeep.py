import numpy as np

class CreateBeep:

    def __init__(self, freq_hz):
        self.freq_hz = freq_hz

    def GetAudioData(self):
        sps = 44100
        duration_s = 0.2
        each_sample_number = np.arange(duration_s * sps)
        waveform = np.sin(2 * np.pi * each_sample_number * self.freq_hz / sps)
        waveform_quiet = waveform * 0.3
        return np.int16(waveform_quiet * 32767)
        

