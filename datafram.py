import torch
from torch.utils.data import Dataset
import torchaudio
import os

# If equals -> one second equals 'one audio'
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050
BATCH_SIZE = 128

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Device is {device}")


class dataset(Dataset):
    def __init__(self, audio_dir, transformetion, target_sample_rate, num_samples, device):
        self.audio_dir = audio_dir
        self.device = device
        self.transformetion = transformetion.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

        self.classes = sorted(os.listdir(audio_dir))
        self.audio_files = self._get_audio_files()

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index):
        audio_file = self.audio_files[index]
        class_label = os.path.basename(os.path.dirname(audio_file))
        signal, sr = torchaudio.load(audio_file)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self.transformetion(signal)
        return signal, class_label

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _cut_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _get_audio_files(self):
        audio_files = []
        for class_dir in self.classes:
            files = os.listdir(os.path.join(self.audio_dir, class_dir))
            audio_files.extend([os.path.join(self.audio_dir, class_dir, file) for file in files])
        return audio_files
