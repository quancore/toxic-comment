import warnings
import random
from pathlib import Path

warnings.filterwarnings("ignore")

import time
from datetime import datetime
import numpy as np
import pandas as pd

from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule

import torch
from torch.utils.data import Dataset

import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

from metrics_loss import AverageMeter, RocAucMeter, ThresholdMeter
import albumentation as albm
from utility import regular_encode


def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec

class DatasetRetriever(Dataset):

    def __init__(self, tokenizer, labels_or_ids, comment_texts, langs, open_subtitles_path=None, maxlen=512, use_train_transforms=False, test=False):
        self.test = test
        self.labels_or_ids = labels_or_ids
        self.comment_texts = comment_texts
        self.langs = langs
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.use_train_transforms = use_train_transforms
        if self.use_train_transforms:
            train_transformations = [('exc_mention', 0.95), ('exc_url', 0.95), ('exc_num', 0.95), ('exc_hashtag', 0.95), ('exc_duplicate', 0.95)]
            self.train_trans_chain = albm.get_transforms(train_transformations)
        
            synthesic_transformations = [('synthesic', 0.5)]
            self.synthesic_trans_chain = albm.get_transforms(synthesic_transformations, open_subtitles_path=open_subtitles_path)
            
            shuffle_transformations = [('shuffle', 1)]
            self.shuffle_trans_chain = albm.get_transforms(shuffle_transformations)

    def __len__(self):
        return self.comment_texts.shape[0]

    def __getitem__(self, idx):
        text = self.comment_texts[idx]
        lang = self.langs[idx]
        if self.test is False:
            label = self.labels_or_ids[idx]
            target = onehot(2, label)

        if self.use_train_transforms:
            text, _ = self.train_trans_chain(data=(text, lang))['data']
            tokens, attention_mask = regular_encode(str(text), self.tokenizer, maxlen=self.maxlen, return_mask=True, batch_encode=False)
            token_length = sum(attention_mask)
            if token_length > 0.8*self.maxlen:
                text, _ = self.shuffle_trans_chain(data=(text, lang))['data']
            elif token_length < 60:
                text, _ = self.synthesic_trans_chain(data=(text, label))['data']
            else:
                tokens, attention_mask = torch.tensor(tokens), torch.tensor(attention_mask)
                return target, tokens, attention_mask
        
        tokens, attention_mask = regular_encode(str(text), self.tokenizer, maxlen=self.maxlen, return_mask=True, batch_encode=False)
        tokens, attention_mask = torch.tensor(tokens), torch.tensor(attention_mask)

        if self.test is False:
            return target, tokens, attention_mask
        
        return self.labels_or_ids[idx], tokens, attention_mask

    def get_labels(self):
        return list(np.char.add(self.labels_or_ids.astype(str), self.langs))

class TPUFitter:
    
    def __init__(self, model, device, config, base_model_path='/', model_name='unnamed', model_version='v1', out_path='/', log_path='/'):
        self.config = config
        self.epoch = 0
        self.base_model_path = base_model_path
        self.model_name = model_name
        self.model_version = model_version
        self.model_path = Path(self.base_model_path, self.model_name, self.model_version)
        self.log_path = Path(log_path, 'log').with_suffix('.txt')
        self.out_path = out_path
        self.node_path = Path(self.out_path, 'node_submissions')
        self.create_dir_structure()

        self.model = model
        self.device = device
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        self.optimizer = AdamW(optimizer_grouped_parameters, lr=config.lr*xm.xrt_world_size())
        self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)

        self.criterion = config.criterion
        self.log(f'Fitter prepared. Device is {self.device}', direct_out=True)
    
    def create_dir_structure(self):
        self.node_path.mkdir(parents=True, exist_ok=True)
        self.log(f'**** Directory structure created ****', direct_out=True)

    def fit(self, train_loader, validation_loader):
        self.log(f'**** Fitting process has been started ****', direct_out=True)
        for e in range(self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR: {lr} \nEpoch:{e}')

            t = time.time()
            para_loader = pl.ParallelLoader(train_loader, [self.device])
            losses, final_scores = self.train_one_epoch(para_loader.per_device_loader(self.device), e)
            
            self.log(f'[RESULT]: Train. Epoch: {self.epoch}, loss: {losses.avg:.5f}, final_score: {final_scores.avg:.5f}, time: {(time.time() - t):.5f}')

            t = time.time()
            para_loader = pl.ParallelLoader(validation_loader, [self.device])
            losses, final_scores, threshold = self.validation(para_loader.per_device_loader(self.device))

            self.log(f'[RESULT]: Validation. Epoch: {self.epoch}, loss: {losses.avg:.5f}, final_score: {final_scores.avg:.5f}, best_th: {threshold.find:.3f}, time: {(time.time() - t):.5f}')
            if self.config.validation_scheduler:
                # self.scheduler.step(metrics=final_scores.avg)
                self.scheduler.step()

            self.epoch += 1
    
    def run_tuning_and_inference(self, test_loader, validation_tune_loader, epoch):
        self.run_validation_tuning(validation_tune_loader, epoch)
        para_loader = pl.ParallelLoader(test_loader, [self.device])
        self.run_inference(para_loader.per_device_loader(self.device))
    
    def run_validation_tuning(self, validation_tune_loader, epoch):
        self.optimizer.param_groups[0]['lr'] = self.config.lr*xm.xrt_world_size() / (epoch + 1)
        para_loader = pl.ParallelLoader(validation_tune_loader, [self.device])
        losses, final_scores = self.train_one_epoch(para_loader.per_device_loader(self.device), epoch)
    
    def validation(self, val_loader):
        self.model.eval()
        losses = AverageMeter()
        final_scores = RocAucMeter()
        threshold = ThresholdMeter()

        t = time.time()
        for step, (targets, inputs, attention_masks) in enumerate(val_loader):
            self.log(
                f'Valid Step {step}, loss: ' + \
                f'{losses.avg:.5f}, final_score: {final_scores.avg:.5f}, ' + \
                f'time: {(time.time() - t):.5f}', step=step
            )
            with torch.no_grad():
                inputs = inputs.to(self.device, dtype=torch.long) 
                attention_masks = attention_masks.to(self.device, dtype=torch.long) 
                targets = targets.to(self.device, dtype=torch.float) 

                outputs = self.model(inputs, attention_masks)
                loss = self.criterion(outputs, targets)
                
                batch_size = inputs.size(0)

                final_scores.update(targets, outputs)
                losses.update(loss.detach().item(), batch_size)
                threshold.update(targets, outputs)
        
        return losses, final_scores, threshold

    def train_one_epoch(self, train_loader, epoch):
        self.log(f'**** Epoch training has started: {epoch} ****', direct_out=True)
        self.model.train()

        losses = AverageMeter()
        final_scores = RocAucMeter()
        t = time.time()
        for step, (targets, inputs, attention_masks) in enumerate(train_loader):
            self.log(
                f'Train Step {step}, loss: ' + \
                f'{losses.avg:.5f}, final_score: {final_scores.avg:.5f}, ' + \
                f'time: {(time.time() - t):.5f}', step=step
            )

            inputs = inputs.to(self.device, dtype=torch.long)
            attention_masks = attention_masks.to(self.device, dtype=torch.long)
            targets = targets.to(self.device, dtype=torch.float)

            self.optimizer.zero_grad()

            outputs = self.model(inputs, attention_masks)
            loss = self.criterion(outputs, targets)

            batch_size = inputs.size(0)
            
            final_scores.update(targets, outputs)
            losses.update(loss.detach().item(), batch_size)

            loss.backward()
            xm.optimizer_step(self.optimizer)

            if self.config.step_scheduler:
                self.scheduler.step()
        
        self.model.eval()
        self.save()
        
        return losses, final_scores

    def run_inference(self, test_loader):
        self.model.eval()
        result = {'id': [], 'toxic': []}
        
        t = time.time()
        for step, (ids, inputs, attention_masks) in enumerate(test_loader):
            self.log(f'Prediction Step {step}, time: {(time.time() - t):.5f}', step=step)

            with torch.no_grad():
                inputs = inputs.to(self.device, dtype=torch.long) 
                attention_masks = attention_masks.to(self.device, dtype=torch.long)
                outputs = self.model(inputs, attention_masks)
                toxics = nn.functional.softmax(outputs, dim=1).data.cpu().numpy()[:,1]

            result['id'].extend(ids.cpu().numpy())
            result['toxic'].extend(toxics)

        result = pd.DataFrame(result)
        print(f'Node path is: {self.node_path}')
        node_count = len(list(self.node_path.glob('*.csv')))
        result.to_csv(self.node_path/f'submission_{node_count}_{datetime.utcnow().microsecond}_{random.random()}.csv', index=False)

    def run_pseudolabeling(self, test_loader, epoch):
        losses = AverageMeter()
        final_scores = RocAucMeter()

        self.model.eval()
        
        t = time.time()
        for step, (ids, inputs, attention_masks) in enumerate(test_loader):

            inputs = inputs.to(self.device, dtype=torch.long) 
            attention_masks = attention_masks.to(self.device, dtype=torch.long)
            outputs = self.model(inputs, attention_masks)
            # print(f'Inputs: {inputs} size: {inputs.size()}')
            # print(f'outputs: {outputs} size: {outputs.size()}')
            toxics = nn.functional.softmax(outputs, dim=1)[:,1]
            toxic_mask = (toxics<=0.4) | (toxics>=0.8)
            # print(attention_masks.size())
            toxics = toxics[toxic_mask]
            inputs = inputs[toxic_mask]
            attention_masks = attention_masks[toxic_mask]
            # print(f'toxics: {toxics.size()}')
            # print(f'inputs: {inputs.size()}')
            if toxics.nelement() != 0:
                targets_int = (toxics>self.config.pseudolabeling_threshold).int()
                targets = torch.stack([onehot(2, target) for target in targets_int])
                # print(targets_int)
                
                self.model.train()
                self.log(
                    f'Pseudolabeling Step {step}, loss: ' + \
                    f'{losses.avg:.5f}, final_score: {final_scores.avg:.5f}, ' + \
                    f'time: {(time.time() - t):.5f}', step=step
                )
    
                targets = targets.to(self.device, dtype=torch.float)
    
                self.optimizer.zero_grad()
    
                outputs = self.model(inputs, attention_masks)
                loss = self.criterion(outputs, targets)
    
                batch_size = inputs.size(0)
                
                final_scores.update(targets, outputs)
                losses.update(loss.detach().item(), batch_size)
    
                loss.backward()
                xm.optimizer_step(self.optimizer)
    
                if self.config.step_scheduler:
                    self.scheduler.step()
    
        self.log(f'[RESULT]: Pseudolabeling. Epoch: {epoch}, loss: {losses.avg:.5f}, final_score: {final_scores.avg:.5f}, time: {(time.time() - t):.5f}')

    def get_submission(self, out_dir):
        submission = pd.concat([pd.read_csv(path) for path in (out_dir/'node_submissions').glob('*.csv')]).groupby('id').mean()
        return submission
    
    def save(self, path=None):
        self.model_path.mkdir(parents=True, exist_ok=True)
        if path is None:
            path = self.model_path/'last-checkpoint.bin'
        
        xm.save(self.model.state_dict(), path)

    def log(self, message, step=None, direct_out=False):
        if direct_out or self.config.verbose:
            if direct_out or step is None or (step is not None and step % self.config.verbose_step == 0):
                xm.master_print(message)
                with open(self.log_path, 'a+') as logger:
                    xm.master_print(f'{message}', logger)


class ToxicSimpleNNModel(nn.Module):

    def __init__(self, transformer):
        super(ToxicSimpleNNModel, self).__init__()
        self.backbone = transformer
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(
            in_features=self.backbone.pooler.dense.out_features*2,
            out_features=2,
        )

    def forward(self, input_ids, attention_masks):
        bs, seq_length = input_ids.shape
        seq_x, _ = self.backbone(input_ids=input_ids, attention_mask=attention_masks)
        apool = torch.mean(seq_x, 1)
        mpool, _ = torch.max(seq_x, 1)
        x = torch.cat((apool, mpool), 1)
        x = self.dropout(x)
        return self.linear(x)