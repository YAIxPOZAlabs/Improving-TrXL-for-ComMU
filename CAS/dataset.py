from torch.utils.data import Dataset
import numpy as np
import torch

def BPM(n): #총 10개의 label
    n = (n - 560)//4
    if n == 10:
        n = 9
    return n

def KEY(n): #총 5개의 label
    n = (n - 601)//5
    return n

def TIMESIGNATURE(n):#총 4개
    return n-626

def PITCHRANGE(n): #총 8개
    return n - 630

def NUMBEROFMEASURE(n): #총 3개
    return n-638


def INSTRUMENT(n): #총9개 
    return n - 641

def GENRE(n):
    return n-650

def MINVELOCITY(n):
    return (n-653)//5

def MAXVELOCITY(n):
    n = (n-653)//5
    if n == 13:
        n = 12
    return n

def TRACKROLE(n):
    return n-719

def RHYTHM(n):
    return n-726
meta_list = [BPM,KEY,TIMESIGNATURE,PITCHRANGE,NUMBEROFMEASURE,
             INSTRUMENT,GENRE,MINVELOCITY,MAXVELOCITY,TRACKROLE,
             RHYTHM]

class Commu_real(Dataset): #given npy that has changing length per sample like GT data
    def __init__(self, meta_npy, midi_npy,seq_len):
        self.meta_npy = meta_npy
        self.midi_npy = midi_npy
        self.seq_len = seq_len
        self.label_npy = np.zeros_like(self.meta_npy)
        for i in range(11):
            self.label_npy[:,i] = np.array(list(map(meta_list[i],meta_npy[:,i])))

    def __len__(self):
        return len(self.meta_npy)

    def __getitem__(self, idx):
        label = self.label_npy[idx]
        midi = np.zeros((1,self.seq_len))
        midi_real = self.midi_npy[idx][:self.seq_len]
        midi[:,:len(midi_real)] = midi_real
        midi = torch.tensor(midi.tolist(),dtype=torch.float)
        label = torch.tensor(label.tolist(),dtype=torch.float)
        return midi,label
    
class Commu_fake(Dataset):#given npy that has fixed length per sample like truncated generated data
    def __init__(self, meta_npy, midi_npy,seq_len):
        self.meta_npy = meta_npy
        self.midi_npy = midi_npy
        self.seq_len = seq_len
        self.label_npy = np.zeros_like(self.meta_npy)
        for i in range(11):
            self.label_npy[:,i] = np.array(list(map(meta_list[i],meta_npy[:,i])))

    def __len__(self):
        return len(self.meta_npy)

    def __getitem__(self, idx):
        label = self.label_npy[idx]
        midi = self.midi_npy[idx]
        # midi = np.zeros((1,self.seq_len))
        # midi_real = self.midi_npy[idx][:self.seq_len]
        # midi[:,:len(midi_real)] = midi_real
        midi = torch.tensor(midi.tolist(),dtype=torch.float).view(1,self.seq_len)
        label = torch.tensor(label.tolist(),dtype=torch.float)
        return midi,label