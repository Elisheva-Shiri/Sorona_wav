import torch
from pathlib import Path
import nussl
import os,fnmatch
import numpy as np
import librosa


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Device is {device}")

class dataset_v3(nussl.datasets.BaseDataset):
    # *args == pass arguments as is to parent
    # *kwargs == pass arguments by name to parent 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_files(self, path):
        lables = {}
        # read all teh files
        #get into the folder
        for folder in os.scandir(path):
            lable = []
            for file in os.scandir(folder):
                lable.append(file)
            #bring the name itself and not the path
            lables[folder] = lable
        return lables

    def _set_zip(self, lables):
        # * able to open a list and pass it as a sepered values for dict
        return list(zip(*lables.values()))
    
    def get_items(self, path):
        lables = self._get_files(path)
        self.lables = list(lables.keys())
        return self._set_zip(lables)
        

    # def _duretion_addapt(self, sources, duretion_min, max_length):
    #     data = []
    #     # Align the audio signals to the maximum length
    #     data = [
    #         np.pad(source.audio_data[0], (0, max_length - source.signal_length), 'constant')
    #         for source in sources
    #     ]

    #     return data
    
    def _compute_ideal_binary_mask(self, source_magnitudes):
        ibm = (source_magnitudes == np.max(source_magnitudes, axis=-1, keepdims=True)).astype(float)
        ibm = ibm / np.sum(ibm, axis=-1, keepdims=True)
        ibm[ibm <= .5] = 0
        return ibm

    def process_item(self, item):
        # TODO: make channle and duretion teh same
        sources  = {}
        duretions = []
        lengths = []
        for file_path in item:
            file = nussl.AudioSignal(file_path.path, sample_rate = self.sample_rate)
            file.to_mono(overwrite=True)
            sources[file_path.path] = file
            duretions.append(file.signal_duration)
            lengths.append(file.signal_length)
        
        # duretion_min = min(duretions)
        
        # # Find the maximum length among the audio signals
        # max_length = max([source.signal_length for source in sources.values()])

        # mix = sum(self._duretion_addapt(sources.values(), duretion_min, max_length)) # sum signals

        metadata = {
            'lables': self.lables
        }
        

        output = {
            'mix': sum(sources.values()),
            'sources': sources,
            'metadata': metadata
        }
        
        return output
    