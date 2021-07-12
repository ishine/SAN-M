import numpy as np
from torch._C import dtype
import oneflow.experimental as flow
import oneflow.experimental.nn as nn
from oneflow.python.ops.array_ops import dynamic_reshape
from oneflow.python.utils.data import DataLoader

from model.pad_mask_utils import IGNORE_ID, pad_list


class AudioTextDataset(nn.Module):
    def __init__(self, 
                 manifest_filepath,
                 batch_seconds,
                 for_training = True
                  ):
        super(AudioTextDataset, self).__init__()
        self.manifest_filepath = manifest_filepath
        self.batch_seconds = batch_seconds
        self.wav_text_duration_list = []
        self.mini_batch_list = []

        with open(self.manifest_filepath, "r") as r_f:
            for line in r_f:
                items = line.strip().split(",")
                wav_file = items[0]
                text = items[1] if for_training else '大'*20

                duration = len(text.replace('@','').replace(' ',''))
            
                self.wav_text_duration_list.append((wav_file, text, duration))


        sorted_data = sorted(self.wav_text_duration_list, key=lambda x: x[2], reverse=True)
        sum_duration = 0
        mini_batch = []
        for (wav_file, text, duration) in sorted_data:
            sum_duration += duration
            if sum_duration <= batch_seconds:
                mini_batch.append((wav_file, text, duration))
            else:
                self.mini_batch_list.append(mini_batch)
                mini_batch = [(wav_file, text, duration)]
                sum_duration = duration
        if len(mini_batch) > 0:
            self.mini_batch_list.append(mini_batch)
    
        
    def __getitem__(self, index):
        return index, self.mini_batch_list[index]

    def __len__(self):
        return len(self.mini_batch_list)



class AudioTextDataLoader(DataLoader):
    def __init__(self, dataset, text_tokenizer, feature_computer):
        super(AudioTextDataLoader, self).__init__(dataset=dataset,
                                                  batch_size=1,
                                                  num_workers=0,
                                                  shuffle=True
                                                  )

        self.text_tokenizer = text_tokenizer
        self.feature_computer = feature_computer


        self.collate_fn = LFRCollate(feature_computer=self.feature_computer, text_tokenizer=self.text_tokenizer)


class LFRCollate(object):
    def __init__(self, feature_computer, text_tokenizer):
        self.feature_computer = feature_computer
        self.text_tokenizer = text_tokenizer


    def __call__(self, batch):
        assert len(batch) == 1
        batch_index, wav_text_duration_list = batch[0]
        xs = []
        ys = []
        wav_paths = []
        batch_total_seconds = 0
        
        for wav_path, text, duration in wav_text_duration_list:
            batch_total_seconds += duration

            wav_feature = self.feature_computer.computer_feature(wav_path)  # 维度:(x,800)

            token_ids = self.text_tokenizer.text_to_tokens(text)                #列表
            xs.append(wav_feature)
            ys.append(token_ids)
            wav_paths.append(wav_path) 

        # x_lens = torch.from_numpy(np.array([x.shape[0] for x in xs]))     #一个batch中所有的帧长度
        # xs_pad = pad_list([torch.from_numpy(x).float() for x in xs], 0)   #特征, 进行补0操作
        # ys_pad = pad_list([torch.LongTensor(y) for y in ys], IGNORE_ID)   #字典，补-1操作
        
        x_lens = flow.Tensor([x.shape[0] for x in xs],dtype = flow.int32)
        xs_pad = pad_list([ flow.Tensor(x,dtype=flow.float32) for x in xs],0)
        ys_pad = pad_list([ flow.Tensor(y,dtype=flow.int32) for y in ys],IGNORE_ID)

        return xs_pad, x_lens, ys_pad, wav_paths
