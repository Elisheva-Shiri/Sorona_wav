import torch
from pathlib import Path
import nussl
import os,fnmatch
import numpy as np
import librosa
from itertools import product



if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Device is {device}")

class dataset_v3(nussl.datasets.BaseDataset):
    # *args == pass arguments as is to parent
    # *kwargs == pass arguments by name to parent 
    def __init__(self, *args, min_product=5, **kwargs):
        self.min_product = min_product
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
            lables[folder.name] = lable[:1000]
        return lables


    # def _set_zip(self, labels):
    #     file_lists = list(labels.values()) 
    #     min_length = min(len(lst) for lst in file_lists)  # Find the minimum length among the file lists
    #     min_length = min(min_length, self.min_product) # don't create product larger than min_prodcut^3
    #     file_lists = [file_list[:min_length] for file_list in file_lists]
    #     mixed_values = list(product(*file_lists))  # Generate the Cartesian product with repeat=min_length
    #     return mixed_values

    def _set_zip(self, lables):
        # * able to open a list and pass it as a sepered values for dict
        return list(zip(*lables.values()))



    
    def get_items(self, path):
        lables = self._get_files(path)
        self.lables = list(lables.keys())
        return self._set_zip(lables)
        
    
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


        metadata = {
            'lables': self.lables
        }

        output = {
            'mix': nussl.AudioSignal(
                audio_data_array=sum(sources.values()).audio_data, 
                sample_rate=self.sample_rate
            ),
            'sources': sources,
            'metadata': metadata
        }
        
        return output
    