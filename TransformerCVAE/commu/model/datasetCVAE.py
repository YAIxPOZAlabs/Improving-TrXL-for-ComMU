import numpy as np
import torch
from commu.preprocessor.encoder.event_tokens import TOKEN_OFFSET
from commu.model.dataset import BaseVocab

class ComMUDatasetCVAE:
    def __init__(self, data_dir, cfg):
        """Load the music corpus
        Args:
            data_dir: The base folder of the preprocessed music dataset
        """
        self._vocab = BaseVocab()
        self._bar_id = 2

        self._train_data = self.load_cache_data(data_dir, "train")
        self._valid_data = self.load_cache_data(data_dir, "valid")
        self._test_data = self.load_cache_data(data_dir, "test")
        self.cfg = cfg

        # Insert start tokens
        # Pad token is added at get_iterator
        # print("USING PAD TOKEN AS START!")
        insert_token = self._vocab.pad_id  # pad as a start token
        self._train_data = [
            torch.from_numpy(arr)
            for arr in self._train_data
        ]
        self._valid_data = [
            torch.from_numpy(arr)
            for arr in self._valid_data
        ]
        self._test_data = [
            torch.from_numpy(arr)
            for arr in self._test_data
        ]

        self._train_seq_length = np.array(
            [ele.shape[0] for ele in self._train_data], dtype=np.int32
        )
        self._valid_seq_length = np.array(
            [ele.shape[0] for ele in self._valid_data], dtype=np.int32
        )
        self._test_seq_length = np.array(
            [ele.shape[0] for ele in self._test_data], dtype=np.int32
        )
        print(
            "Loaded Data, #Samples Train/Val/Test:{}/{}/{}".format(
                len(self._train_data), len(self._valid_data), len(self._test_data)
            )
        )
        print(
            "             #Avg Length:{}/{}/{}".format(
                np.mean([len(ele) for ele in self._train_data]),
                np.mean([len(ele) for ele in self._valid_data]),
                np.mean([len(ele) for ele in self._test_data]),
            )
        )
        print(
            "             #Total Number of Valid/Test Tokens: {}/{}".format(
                (self._valid_seq_length - 1).sum(), (self._test_seq_length - 1).sum()
            )
        )

    def load_cache_data(self, dir_name, mode):
        if mode == "train":
            data_input = np.load(dir_name + '/input_train.npy', allow_pickle=True)
            data_target = np.load(dir_name + '/target_train.npy', allow_pickle=True)
            dat = []
            for i in range(len(data_input)):
                dat.append(np.concatenate((np.array(data_input[i], dtype=int), data_target[i])))
        else:
            data_input = np.load(dir_name + '/input_val.npy', allow_pickle=True)
            data_target = np.load(dir_name + '/target_val.npy', allow_pickle=True)
            dat = []
            for i in range(len(data_input)):
                dat.append(np.concatenate((np.array(data_input[i], dtype=int), data_target[i])))
        return np.array(dat, dtype=object)

    @property
    def vocab(self):
        return self._vocab

    @property
    def train_data(self):
        return self._train_data

    @property
    def valid_data(self):
        return self._valid_data

    @property
    def test_data(self):
        return self._test_data

    @property
    def train_seq_length(self):
        return self._train_seq_length

    @property
    def valid_seq_length(self):
        return self._valid_seq_length

    @property
    def test_seq_length(self):
        return self._test_seq_length
    
    # meta_len means the length of the metadata
    # bptt means target sequence length(same with source sequence length) -> this should be even number
    def get_iterator( 
            self, batch_size, bptt, meta_len, device, split="train", do_shuffle=True, seed=None
    ):
        if split == "train":
            split_data = self.train_data
            split_seq_lengths = self.train_seq_length
        elif split == "valid":
            split_data = self.valid_data
            split_seq_lengths = self.valid_seq_length
        elif split == "test":
            split_data = self.test_data
            split_seq_lengths = self.test_seq_length
        else:
            raise NotImplementedError
        total_sample_num = len(split_data)

        def iterator():
            perm = np.arange(total_sample_num)
            if do_shuffle:
                rng = np.random.RandomState(seed)
                rng.shuffle(perm)
            assert batch_size < total_sample_num
            tracker_list = [(i, meta_len) for i in range(batch_size)]
            next_idx = batch_size
            data = torch.LongTensor(bptt, batch_size)
            target = torch.LongTensor(bptt, batch_size)
            meta = torch.LongTensor(meta_len, batch_size)

            while True:
                # Generate the samples
                # Fill with pad_id
                data[:] = self.vocab.pad_id
                target[:] = self.vocab.pad_id
                batch_token_num = 0
                for i in range(batch_size):
                    idx, pos = tracker_list[i]
                    while idx < total_sample_num:
                        seq_id = perm[idx]
                        seq_length = split_seq_lengths[seq_id]
                        if pos + 1 >= seq_length:
                            idx, pos = next_idx, 0
                            tracker_list[i] = (idx, pos)
                            next_idx += 1
                            continue
                        else:
                            n_new = min(seq_length - pos, bptt)
                            # add start token here!
                            data[1:n_new, i] = split_data[seq_id][pos: pos + n_new - 1]
                            target[:n_new, i] = split_data[seq_id][
                                                (pos): (pos + n_new)]
                            batch_token_num += n_new
                            # find next position here
                            # next data starts from the last bar appeared in current data
                            bar_idx = (target[:, i] == self._bar_id).nonzero().max().item()
                            tracker_list[i] = (idx, pos + bar_idx)
                            meta[:meta_len, i] = split_data[seq_id][:meta_len]
                            break
                if batch_token_num == 0:
                    # Haven't found anything to fill. This indicates we have reached the end
                    if do_shuffle:
                        rng.shuffle(perm)
                    else:
                        return  # One pass dataloader when do_shuffle is False
                    tracker_list = [(i, 0) for i in range(batch_size)]
                    next_idx = batch_size
                    continue

                yield data.to(device), target.to(device), meta.to(device), batch_token_num

        return iterator

    def eval_iterator(
            self, batch_size, bptt, meta_len, device, split="valid", local_rank=0, world_size=0
    ):
        if split == "valid":
            split_data = self.valid_data
            split_seq_lengths = self.valid_seq_length
        elif split == "test":
            split_data = self.test_data
            split_seq_lengths = self.test_seq_length
        else:
            raise NotImplementedError
        if world_size > 0:
            all_sample_num = len(split_data)
            if local_rank == world_size - 1:
                begin_idx = all_sample_num // world_size * local_rank
                end_idx = all_sample_num
            else:
                begin_idx = all_sample_num // world_size * local_rank
                end_idx = all_sample_num // world_size * (local_rank + 1)
            split_data = split_data[begin_idx:end_idx]
            split_seq_lengths = split_seq_lengths[begin_idx:end_idx]
        total_sample_num = len(split_data)

        def iterator():
            data = torch.LongTensor(bptt, batch_size)
            target = torch.LongTensor(bptt, batch_size)
            meta = torch.LongTensor(meta_len, batch_size)
            for batch_begin in range(0, total_sample_num, batch_size):
                batch_end = min(batch_begin + batch_size, total_sample_num)
                max_seq_length = max(split_seq_lengths[batch_begin:batch_end])
                for seq_begin in range(meta_len, max_seq_length - 1, bptt):
                    data[:] = self.vocab.pad_id
                    target[:] = self.vocab.pad_id
                    batch_token_num = 0
                    for i in range(batch_begin, batch_end):
                        if split_seq_lengths[i] > seq_begin + 1:
                            n_new = (
                                    min(seq_begin + bptt, split_seq_lengths[i] - 1)
                                    - seq_begin
                            )
                            data[:n_new, i - batch_begin] = split_data[i][
                                                            seq_begin: seq_begin + n_new
                                                            ]
                            target[:n_new, i - batch_begin] = split_data[i][
                                                              (seq_begin + 1): (seq_begin + n_new + 1)
                                                              ]
                            batch_token_num += n_new
                            meta[:, i - batch_begin] = split_data[i][:meta_len]

                    yield data.to(device), target.to(device), meta.to(device), batch_token_num

        return iterator
