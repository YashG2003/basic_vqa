import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from utils import text_helper    # read about text_helper
# Read from torch.utils.data documentation
# A map-style dataset is one that implements the __getitem__() and __len__() protocols, and represents a map from (possibly non-integral) indices/keys to data samples.
# For example, such a dataset, when accessed with dataset[idx], could read the idx-th image and its corresponding label from a folder on the disk.

class VqaDataset(data.Dataset):

    def __init__(self, input_dir, input_vqa, max_qst_length=30, max_num_ans=10, transform=None):
        self.input_dir = input_dir
        self.vqa = np.load(input_dir+'/'+input_vqa)
        self.qst_vocab = text_helper.VocabDict(input_dir+'/vocab_questions.txt')  # the data loaded goes into the file in directory vocab_questions
        self.ans_vocab = text_helper.VocabDict(input_dir+'/vocab_answers.txt')
        self.max_qst_length = max_qst_length
        self.max_num_ans = max_num_ans
        self.load_ans = ('valid_answers' in self.vqa[0]) and (self.vqa[0]['valid_answers'] is not None)  # self.load_ans is of bool type
        self.transform = transform # read about it

    def __getitem__(self, idx):

        vqa = self.vqa
        qst_vocab = self.qst_vocab
        ans_vocab = self.ans_vocab
        max_qst_length = self.max_qst_length
        max_num_ans = self.max_num_ans
        transform = self.transform
        load_ans = self.load_ans

        image = vqa[idx]['image_path']    # idx is the id of batch of images
        image = Image.open(image).convert('RGB')  # every image has specific RGB intensity
        qst2idc = np.array([qst_vocab.word2idx('<pad>')] * max_qst_length)  # padded with '<pad>' in 'ans_vocab'
        qst2idc[:len(vqa[idx]['question_tokens'])] = [qst_vocab.word2idx(w) for w in vqa[idx]['question_tokens']]    # read 
        sample = {'image': image, 'question': qst2idc}
        # sample is a dictionary containing images and question corresponding to it

        if load_ans:  # load_ans is of bool type
            ans2idc = [ans_vocab.word2idx(w) for w in vqa[idx]['valid_answers']]
            ans2idx = np.random.choice(ans2idc)
            sample['answer_label'] = ans2idx         # for training

            mul2idc = list([-1] * max_num_ans)       # padded with -1 (no meaning) not used in 'ans_vocab'
            mul2idc[:len(ans2idc)] = ans2idc         # our model should not predict -1
            sample['answer_multi_choice'] = mul2idc  # for evaluation metric of 'multiple choice'

        if transform:
            sample['image'] = transform(sample['image'])

        return sample
    # we have images,question and its answer from the dataset now

    def __len__(self):

        return len(self.vqa)


def get_loader(input_dir, input_vqa_train, input_vqa_valid, max_qst_length, max_num_ans, batch_size, num_workers):

    transform = {
        phase: transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.485, 0.456, 0.406),
                                                        (0.229, 0.224, 0.225))]) 
        for phase in ['train', 'valid']}

    vqa_dataset = {
        'train': VqaDataset(
            input_dir=input_dir,
            input_vqa=input_vqa_train,
            max_qst_length=max_qst_length,
            max_num_ans=max_num_ans,
            transform=transform['train']),
        'valid': VqaDataset(
            input_dir=input_dir,
            input_vqa=input_vqa_valid,
            max_qst_length=max_qst_length,
            max_num_ans=max_num_ans,
            transform=transform['valid'])}

    data_loader = {
        phase: torch.utils.data.DataLoader(
            dataset=vqa_dataset[phase],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers)
        for phase in ['train', 'valid']}

    return data_loader
