import torch
from torch import nn
import torchaudio
from torch.utils.data import Dataset

class GaussianNoise(nn.Module):    
    def __init__(self, mean=0, std=0.05):
        super(GaussianNoise, self).__init__()
        
        self.noiser = torch.distributions.Normal(mean, std)   
            
    def forward(self, wav):
        wav = wav + self.noiser.sample(wav.size())     
        wav = wav.clamp(-1, 1)
        
        return wav
    
class YoutubeNoise(nn.Module):    
    def __init__(self, alpha=0.05):
        super(YoutubeNoise, self).__init__()
        
        self.alpha = alpha
        self.noise_wav = youtube_noise
            
    def forward(self, wav):
        wav = wav + self.alpha * self.noise_wav[:wav.shape[-1]] 
        wav = wav.clamp(-1, 1)
        
        return wav

class SpeechCommands(Dataset):
    def __init__(self, config, X, y, train=True):
        self.paths = X
        self.labels = y
        self.train = train
        self.config = config
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img = torch.zeros(1, self.config.melspec_n_mels, self.config.img_padding_length)
        wav, sr = torchaudio.load(self.paths[idx])
        
        if self.train:
            wav_proc = nn.Sequential(GaussianNoise(0, 0.01),
                                    torchaudio.transforms.MelSpectrogram(sample_rate=self.config.melspec_sample_rate, 
                                                                        n_mels=self.config.melspec_n_mels, 
                                                                        n_fft=self.config.melspec_n_fft, 
                                                                        hop_length=self.config.melspec_hop_length, 
                                                                        f_max=self.config.melspec_f_max),
                                    torchaudio.transforms.FrequencyMasking(freq_mask_param=self.config.specaug_freq_mask_param),
                                    torchaudio.transforms.TimeMasking(time_mask_param=self.config.specaug_time_mask_param)
                                    )
        else:
            wav_proc = nn.Sequential(torchaudio.transforms.MelSpectrogram(sample_rate=self.config.melspec_sample_rate, 
                                                                        n_mels=self.config.melspec_n_mels, 
                                                                        n_fft=self.config.melspec_n_fft, 
                                                                        hop_length=self.config.melspec_hop_length, 
                                                                        f_max=self.config.melspec_f_max),
                                    )
            
        mel_spectrogram = torch.log(wav_proc(wav) + 1e-9)
        img[0, :, :mel_spectrogram.size(2)] = mel_spectrogram
        
        return img.reshape(self.config.melspec_n_mels, self.config.img_padding_length), self.labels[idx]