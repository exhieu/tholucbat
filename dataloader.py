#%%
import os
import torch
from torch.utils.data.dataset import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from filelock import FileLock
from transformers.utils import logging
from typing import Dict, List, Optional
import pickle
import random
import time
logger = logging.get_logger(__name__)

class PoemDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    Parameters:
    ----------
    tokenizers : is pretrain tokenizer of PhoBERT
    file_path  : path to file train, test
    block_size : size of 1 block , optinal
    cache_dir  : just load 1 once and saved

    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        file_path: str,
        block_size: int,
        overwrite_cache=False,
        cache_dir: Optional[str] = None,
    ):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else directory,
            "cached_lm_{}_{}_{}".format(
                tokenizer.__class__.__name__,
                str(block_size),
                filename,
            ),
        )

        # -----------Make sure only the first process in distributed training processes the dataset,----------------#
        # ---------------------------------------and the others will use the cache------------------------#
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )

            else:
                logger.info(f"Creating features from dataset file at {directory}")

                self.examples = []
                with open(file_path, encoding="utf-8") as f:
                    text = f.read()
                #-----convert text to tokenizers----------------------------#
                '''
                1. Convert word -> subword (tokenizer.tokenize(text))
                2. COnvert subword -> number (tokenizer.convert_tokens_to_ids)
                '''
                tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

                # ------------- Truncate in block of block_size-----------------#
                #-----------Beacuse add_token('\n') -> inds = 64001------------#
                #--------If len(block_size)>56 so cut and add_special_tokens (<s>, </s>)---------------#
                i = 0
                while i < len(tokenized_text) - block_size + 1:
                    inds = tokenized_text[i : i + block_size]
                    for j in range(0, len(inds)):
                        if inds[j]==64001:
                            inds = inds[j+1:] #remove the first \n
                            break
                    for j in range(len(inds)-1, 0, -1):
                        if inds[j]==64001:
                            inds = inds[:j-1] #remove \n
                            break
                    i += len(inds)
                    self.examples.append(
                        tokenizer.build_inputs_with_special_tokens(inds)
                    )
                    
                # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
                # If your dataset is small, first you should loook for a bigger one :-) and second you
                # can change this behavior by adding (model specific) padding.

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)
 #-----------Load dataset-----------------------#
from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling, LineByLineWithSOPTextDataset

def load_dataset(train_path, test_path, custokenizer):
    train_dataset = PoemDataset(
          tokenizer=custokenizer,
          file_path=train_path,
          block_size= 56)#256
     
    test_dataset = PoemDataset(
          tokenizer=custokenizer,
          file_path=test_path,
          block_size=56)   
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=custokenizer, mlm=False,
    )
    return train_dataset,test_dataset,data_collator
#%%
train_dataset,test_dataset,data_collator = load_dataset('data_train_process.txt','data_test_process.txt',custokenizer)
#-----------Test dataloader----------------#
print(len(test_dataset))
print(len(train_dataset))
#-------------Test decode to sentence ---------------#
print(custokenizer.decode(test_dataset[7]))

from transformers import Trainer, TrainingArguments, GPT2Config, GPT2LMHeadModel
#--------------------------Load  pretrain model GPT-2--------------------#
model_gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
# Random weights => fine-turning model
rand_weight = torch.rand(model_gpt2.lm_head.weight.shape)
print(rand_weight)
model_gpt2.lm_head.weight = torch.nn.parameter.Parameter(rand_weight)
'''
Because GPT2 has vocabulary_size 50257 and (wte): Embedding(50257, 768)
So  convert vocabulary_size= 64002, Embedding(64002, 768)
'''
task_gpt2 = {"text-generation": {"do_sample": True, "max_length": 56}} #edit output size
config_gpt2 = configuration = GPT2Config(vocab_size=64002, n_positions=58, n_ctx=58,
                           task_specific_params=task_gpt2,
                           eos_token_id = 2,
                           bos_token_id = 0,
                           pad_token_id = 1,
                           sep_token_id = 2,
                          #  eos_token_id=custokenizer.eos_token_id,
                          #  bos_token_id=custokenizer.bos_token_id, 
                          #  pad_token_id=custokenizer.pad_token_id,
                          #  sep_token_id=custokenizer.sep_token_id
                           )
model_gpt2 = GPT2LMHeadModel(config_gpt2)
model_gpt2
#save model_gpt2 (vocabulary_size =64002)
model_gpt2.save_pretrained('/content/drive/MyDrive/BERT/save_modelGPT2/')
task = {"text-generation": {"do_sample": True, "max_length": 56}} #edit output size
configuration = GPT2Config(vocab_size=64002, n_positions=58, n_ctx=58,
                           task_specific_params=task,
                           eos_token_id = 2,
                           bos_token_id = 0,
                           pad_token_id = 1,
                           sep_token_id = 2,
                          #  eos_token_id=custokenizer.eos_token_id,
                          #  bos_token_id=custokenizer.bos_token_id, 
                          #  pad_token_id=custokenizer.pad_token_id,
                          #  sep_token_id=custokenizer.sep_token_id
                           )
poem = GPT2LMHeadModel(configuration)

# Load weights of model_gpt2 ( random weights)
load_model_gpt2 = GPT2LMHeadModel.from_pretrained('/content/drive/MyDrive/BERT/save_modelGPT2/')
poem.load_state_dict(load_model_gpt2.state_dict())
#-----------Print process training ------------#
from transformers.trainer_callback import TrainerCallback
from transformers import pipeline
class PrinterCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if int(state.epoch)%10==0:
            pipe = pipeline('text-generation', model=model, tokenizer=custokenizer, device=0)
            with open("/content/drive/MyDrive/BERT/sample.txt", "a") as f:
                f.write(pipe('<s> tìm về một thuở hạ xưa')[0]['generated_text'])
                f.write("\n===========================================\n")
                f.close()
training_args = TrainingArguments(
    output_dir="/content/drive/MyDrive/BERT/gpt2-poem", #The output directory
    overwrite_output_dir=True, #overwrite the content of the output directory
    num_train_epochs=100, # number of training epochs
    per_device_train_batch_size=8, # batch size for training  
    per_device_eval_batch_size=16,  # batch size for evaluation
    save_steps=5000, # after # steps model is saved 
    save_total_limit = 2, # delete other checkpoints
    warmup_steps=5000,    # number of warmup steps for learning rate scheduler
    # logging_dir='/content/drive/MyDrive/BERT/gpt2-poem/logs', # directory for storing logs
    logging_steps=5000,
    )


device = torch.device('cuda')
trainer = Trainer(
    model=poem, # GPT2
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    callbacks = [PrinterCallback],
)
# -------Train and save model-----------#
trainer.train()
trainer.save_model()
# %%
