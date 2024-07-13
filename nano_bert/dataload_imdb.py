import torch
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def encode_label(label):
    if label == 'pos':
        return 1
    elif label == 'neg':
        return 0
    raise Exception(f'Unknown Label: {label}!')


class IMDBDataloader:
    def __init__(self, data, test_data, tokenizer, label_encoder, batch_size, val_frac=0.2):
        train_data, val_data = train_test_split(data, shuffle=True, random_state=1, test_size=val_frac)

        self.splits = {
            'train': [d['text'] for d in train_data],
            'test': [d['text'] for d in test_data],
            'val': [d['text'] for d in val_data]
        }

        self.labels = {
            'train': [d['label'] for d in train_data],
            'test': [d['label'] for d in test_data],
            'val': [d['label'] for d in val_data]
        }

        self.tokenized = {
            'train': [tokenizer(record).unsqueeze(0) for record in
                      tqdm(self.splits['train'], desc='Train Tokenization',position=0)], # divide different sentences in comments
            'test': [tokenizer(record).unsqueeze(0) for record in tqdm(self.splits['test'], desc='Test Tokenization',position=0)],
            'val': [tokenizer(record).unsqueeze(0) for record in tqdm(self.splits['val'], desc='Val Tokenization',position=0)],
        }

        self.encoded_labels = {
            'train': [label_encoder(label) for label in tqdm(self.labels['train'], desc='Train Label Encoding',position=0)],
            'test': [label_encoder(label) for label in tqdm(self.labels['test'], desc='Test Label Encoding',position=0)],
            'val': [label_encoder(label) for label in tqdm(self.labels['val'], desc='Val Label Encoding',position=0)],
        }

        self.curr_batch = 0
        self.batch_size = batch_size
        self.iterate_split = None

    def peek(self, split):
        return {
            'input_ids': self.splits[split][self.batch_size * self.curr_batch:self.batch_size * (self.curr_batch + 1)],
            'label_ids': self.labels[split][self.batch_size * self.curr_batch:self.batch_size * (self.curr_batch + 1)],
        }

    def take(self, split):
        batch = self.splits[split][self.batch_size * self.curr_batch:self.batch_size * (self.curr_batch + 1)]
        labels = self.labels[split][self.batch_size * self.curr_batch:self.batch_size * (self.curr_batch + 1)]
        self.curr_batch += 1
        return {
            'input_ids': batch,
            'label_ids': labels,
        }

    def peek_tokenized(self, split):
        return {
            'input_ids': torch.cat(
                self.tokenized[split][self.batch_size * self.curr_batch:self.batch_size * (self.curr_batch + 1)],
                dim=0),
            'label_ids': torch.tensor(
                self.encoded_labels[split][self.batch_size * self.curr_batch:self.batch_size * (self.curr_batch + 1)],
                dtype=torch.long),
        }

    def peek_index_tokenized(self, index, split):
        return {
            'input_ids': torch.cat(
                [self.tokenized[split][index]],
                dim=0),
            'label_ids': torch.tensor(
                [self.encoded_labels[split][index]],
                dtype=torch.long),
        }

    def peek_index(self, index, split):
        return {
            'input_ids': [self.splits[split][index]],
            'label_ids': [self.labels[split][index]],
        }

    def take_tokenized(self, split):
        batch = self.tokenized[split][self.batch_size * self.curr_batch:self.batch_size * (self.curr_batch + 1)]
        labels = self.encoded_labels[split][self.batch_size * self.curr_batch:self.batch_size * (self.curr_batch + 1)]
        self.curr_batch += 1
        return {
            'input_ids': torch.cat(batch, dim=0),
            'label_ids': torch.tensor(labels, dtype=torch.long),
        }

    def get_split(self, split):
        self.iterate_split = split
        return self

    def steps(self, split):
        return len(self.tokenized[split]) // self.batch_size

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        if self.batch_size * self.curr_batch < len(self.splits[self.iterate_split]):
            return self.take_tokenized(self.iterate_split)
        else:
            raise StopIteration

    def reset(self):
        self.curr_batch = 0
        
def mask_text(cmt, vacab_size, max_seq_len, n_sp): # cmt : a comment. n_sp: number of special symbols in tokenizer
    mps = []
    mcmt = cmt.clone()
    unique_elements, counts = torch.unique(cmt, return_counts=True)
    m_range = (max_seq_len - counts[0].item()) * 0.15 # range of masking indice
#     print(m_range, counts[0].item())
    while len(mps) < m_range:
        temp = random.randint(0, max_seq_len - 1)
        if mcmt[temp] > n_sp - 1:
            mps.append(temp)
            mcmt[temp] = 0 # set to mask
    temp_m = random.sample(range(0, len(mps) - 1), 2 * int(m_range * 0.1 + 1)) # fetch some masked words to their original value
    half = int(len(temp_m)/2)
    tmps = torch.tensor(mps) # tensorlize mps to get location of masks to be changed
    mcmt[tmps[temp_m[:half]]] = cmt[temp_m[:half]] # 10% percent original
    mcmt[tmps[temp_m[half:]]] = torch.tensor([random.randint(n_sp, vacab_size - 1) for i in range(half)]).to(device) # 10% percent random vocab
    return mps, mcmt

def spit_cmt(cmt): # split  comment into sentences
    bg = torch.where(cmt==5)[0] # begin at '[SOS]'
    ed = torch.where(cmt==6)[0] # end at '.'
    spit = []
    n_st = len(bg) # number of next sentence prediction task in this comment
    for i in range(n_st):
        sts = cmt[bg[i].item():ed[i].item()+1]
        spit.append(sts)
    return spit, n_st

def pad_nsp(nsp, max_seq_len): # add '[CLS]' and padding on given composed 2 sentences in GPU
#     print('len nsp = ', len(nsp))
    return torch.cat((torch.tensor([0]).to(device), nsp, torch.ones(max_seq_len - len(nsp) - 1).to(device)), 0)

def nsp_gen(spit, n_st, max_seq_len): # generate nsp input      
    nsp = torch.zeros((2 * (n_st - 1), max_seq_len)).to(device)
    if n_st < 3:
        nsp = torch.zeros((1, max_seq_len)).to(device)
#     print(nsp[:10], n_st)
    for id_i in range(n_st - 1):
        nsp[2 * id_i] = pad_nsp(torch.cat((spit[id_i], spit[id_i + 1]), 0), max_seq_len) # a sentence followed by the next
        if n_st > 2:
            id_s = random.randint(0, n_st - 1) # selected id for nsp
            while id_s == id_i + 1 or id_s == id_i: # excluded the next sentence
                id_s = random.randint(0, n_st - 1)
            nsp[2 * id_i + 1] = pad_nsp(torch.cat((spit[id_i], spit[id_s]), 0), max_seq_len) # a sentence not followed by the next
    return nsp.long().to(device)

def plot_results(history, model, do_val=True):
    fig, ax = plt.subplots(figsize=(8, 8))

    x = list(range(0, len(history['train_losses'])))

    # loss

    ax.plot(x, history['train_losses'], label='train_loss')

    if do_val:
        ax.plot(x, history['val_losses'], label='val_loss')

    plt.title(model + ' Train / Validation Loss')
    plt.legend(loc='upper right')
    plt.savefig('pic/' + model + 'Loss')

    # accuracy

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(x, history['train_acc'], label='train_acc')

    if do_val:
        ax.plot(x, history['val_acc'], label='val_acc')

    plt.title(model + ' Train / Validation Accuracy')
    plt.legend(loc='upper left')
    plt.savefig('pic/' + model + ' Accuracy')

    # f1-score

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(x, history['train_f1'], label='train_f1')

    if do_val:
        ax.plot(x, history['val_f1'], label='val_f1')

    plt.title(model + ' Train / Validation F1')
    plt.legend(loc='upper left')
    plt.savefig('pic/' + model + 'Train_Validation Accuracy')

    fig.show()