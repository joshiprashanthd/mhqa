import pytorch_lightning as pl
from torch.utils.data import SequentialSampler,RandomSampler, DataLoader
from torch import nn
import numpy as np
import math
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer,AutoModel
import functools

def compute_accuracy(logits, labels):
  prediction = torch.argmax(logits, axis=-1)
  correct_predictions = torch.sum(prediction==labels)
  accuracy = correct_predictions.cpu().detach().numpy()/prediction.size()[0]
  return accuracy

class MCQAModel(pl.LightningModule):
  def __init__(self,
               model_name_or_path,
               args):
    
    super().__init__()
    self.init_encoder_model(model_name_or_path)
    self.args = args
    self.batch_size = self.args['batch_size']
    self.dropout = nn.Dropout(self.args['hidden_dropout_prob'])
    self.linear = nn.Linear(self.args['hidden_size'], out_features=1)
    self.ce_loss = nn.CrossEntropyLoss()
    self.save_hyperparameters()
  
  def init_encoder_model(self,model_name_or_path):
    self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    self.model: AutoModel = AutoModel.from_pretrained(model_name_or_path)
 
  def prepare_dataset(self,train_dataset, val_dataset, test_dataset=None):
    """
    helper to set the train and val dataset. Doing it during class initialization
    causes issues while loading checkpoint as the dataset class needs to be 
    present for the weights to be loaded.
    """
    self.train_dataset = train_dataset
    self.val_dataset = val_dataset
    if test_dataset != None:
        self.test_dataset = test_dataset
    else:
        self.test_dataset = val_dataset
  
  def forward(self,input_ids,attention_mask,token_type_ids=None):
    outputs = self.model(input_ids=input_ids,
                       attention_mask=attention_mask,
                       token_type_ids=token_type_ids)
    
    pooled_output = outputs[1]
    pooled_output = self.dropout(pooled_output)
    logits = self.linear(pooled_output)
    reshaped_logits = logits.view(-1,self.args['num_choices'])
    return reshaped_logits
  
  def training_step(self, batch, batch_idx):
    inputs, labels = batch
    for key in inputs:
      inputs[key] = inputs[key].to(self.args["device"])
    logits = self(**inputs)
    loss = self.ce_loss(logits,labels)

    accuracy = compute_accuracy(logits, labels)
    self.log("train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
    self.log("train_acc", accuracy, prog_bar=True, logger=True, on_step=True, on_epoch=True)
    return loss
  
  def test_step(self, batch, batch_idx):
    inputs,labels = batch
    for key in inputs:
      inputs[key] = inputs[key].to(self.args["device"])
    logits = self(**inputs)
    loss = self.ce_loss(logits,labels)

    accuracy = compute_accuracy(logits, labels)
    self.log("test_loss", loss, logger=True, on_epoch=True, on_step=True)
    self.log("test_acc", accuracy, logger=True, on_epoch=True, on_step=True)
    return loss

  def validation_step(self, batch, batch_idx):
    inputs, labels = batch
    for key in inputs:
      inputs[key] = inputs[key].to(self.args["device"])
    logits = self(**inputs)
    loss = self.ce_loss(logits,labels)
    accuracy = compute_accuracy(logits, labels)
    self.log("val_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
    self.log("val_acc", accuracy, prog_bar=True, logger=True, on_step=True, on_epoch=True)
    return loss
        
  def configure_optimizers(self):
    optimizer = AdamW(self.parameters(),lr=self.args['learning_rate'],eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=(self.args['num_epochs'] + 1) * math.ceil(len(self.train_dataset) / self.args['batch_size']),
    )
    return [optimizer],[scheduler]
  
  def process_batch(self, batch, tokenizer, max_len=32):
    expanded_batch = []
    labels = []
    for data_tuple in batch:
        question,options,label = data_tuple
        question_option_pairs = [question+' '+option for option in options]
        labels.append(label)
        expanded_batch.extend(question_option_pairs)
    tokenized_batch = tokenizer.batch_encode_plus(expanded_batch, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")

    return tokenized_batch, torch.LongTensor(labels)
  
  def train_dataloader(self):
    train_sampler = RandomSampler(self.train_dataset)
    model_collate_fn = functools.partial(
      self.process_batch,
      tokenizer=self.tokenizer,
      max_len=self.args['max_len']
    )
    train_dataloader = DataLoader(self.train_dataset,
                                batch_size=self.batch_size,
                                sampler=train_sampler,
                                collate_fn=model_collate_fn,
                                num_workers=95)
    return train_dataloader
  
  def val_dataloader(self):
    eval_sampler = SequentialSampler(self.val_dataset)
    model_collate_fn = functools.partial(
      self.process_batch,
      tokenizer=self.tokenizer,
      max_len=self.args['max_len']
    )
    val_dataloader = DataLoader(self.val_dataset,
                                batch_size=self.batch_size,
                                sampler=eval_sampler,
                                collate_fn=model_collate_fn,
                                num_workers=95)
    return val_dataloader
  
  def test_dataloader(self):
    eval_sampler = SequentialSampler(self.test_dataset)
    model_collate_fn = functools.partial(
      self.process_batch,
      tokenizer=self.tokenizer,
      max_len=self.args['max_len']
    )
    test_dataloader = DataLoader(self.test_dataset,
                                batch_size=self.batch_size,
                                sampler=eval_sampler,
                                collate_fn=model_collate_fn, 
                                num_workers=95)
    return test_dataloader