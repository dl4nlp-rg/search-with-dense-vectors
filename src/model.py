from dataset import *
from helpers import *
from modules import *
import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split

class TwoTower(pl.LightningModule):
    """Two tower model for document retrieval using query and document."""
    def __init__(self,
                 learning_rate,
                 queries,
                 doc_queries,
                 triples,
                 batch_size=64,
                 dim=64,
                 epochs = 5,
                 max_length=32,
                 shuffle=True,
                 ):
        super(TwoTower, self).__init__()

        # Configuration
        self.learning_rate = learning_rate
        self.batch_size=batch_size
        self.max_length = max_length
        self.loss = CosineSimilarityLoss()
        self.optimizer = torch.optim.Adam
        self.opt_params = {'lr': learning_rate, 'eps': 1e-08, 'betas': (0.9, 0.999)}
        self.warmup_steps = 500
        self.training_steps = (epochs*len(triples))/batch_size
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)

        #Data
        self.queries = queries
        self.doc_queries = doc_queries
        self.triples = triples

        self.shuffle=shuffle
        if use_cuda:
            self.pin_mem = True
            self.n_workers = 0
        else:
            self.pin_mem = False
            self.n_workers = nproc

        # Models
        self.query_encoder = Encoder(dim)
        self.doc_encoder = Encoder(dim)

    def prepare_data(self):
        train_triples, val_triples = train_test_split(self.triples, train_size=0.8)

        self.train_data = MyDataset(
                triples = train_triples,
                queries = self.queries,
                docs = self.doc_queries,
                max_length = self.max_length,
                tokenizer=self.tokenizer)
        
        self.valid_data = MyDataset(
                triples = val_triples,
                queries = self.queries,
                docs = self.doc_queries,
                max_length = self.max_length,
                tokenizer=self.tokenizer)
    
    @gpu_mem_restore
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size,
                          shuffle=self.shuffle, num_workers=self.n_workers,
                          pin_memory=self.pin_mem)

    @gpu_mem_restore
    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.n_workers,
                          pin_memory=self.pin_mem)
    
    def configure_optimizers(self):
        optimizer = self.optimizer(
            [e for e in self.parameters() if e.requires_grad],
            **self.opt_params)
        return optimizer

    # learning rate warm-up and linear scheduler
    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None):
        # warm up lr
        if self.trainer.global_step < self.warmup_steps:
            lr_scale = min(1., float(self.trainer.global_step + 1) / float(self.warmup_steps))
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.learning_rate
        # linear decrease schedule
        else:
            lr_scale = max(0.0,
                float(self.training_steps - self.trainer.global_step) / float(max(1,
                                                self.training_steps - self.warmup_steps)))
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.learning_rate
        # update params
        optimizer.step()
        optimizer.zero_grad()

    def forward(self,batch):
        q_tok, q_mask, q_type, p_tok, p_mask, p_type, n_tok, n_mask, n_type, query, doc_pos, doc_neg = batch
        
        query_embedding = self.query_encoder(q_tok,q_mask,q_type)
        p_doc_embedding = self.doc_encoder(p_tok,p_mask,p_type)
        n_doc_embedding = self.doc_encoder(n_tok,n_mask,n_type)
        
        loss, sim_pos, sim_neg = self.loss(query_embedding, p_doc_embedding, n_doc_embedding) 

        return loss.mean(), sim_pos, sim_neg

    def _step(self, prefix, batch, batch_nb):

        if prefix == 'train':
            loss, _, _ = self(batch)
            log = {f'{prefix}_loss': loss}
            return {'loss': loss, 'log': log,}

        elif(prefix=='val'):
            loss, sim_pos, sim_neg = self(batch)
            acc = (sim_pos > sim_neg).sum()
            acc = (acc.type(torch.float))/self.batch_size
            log = {f'{prefix}_loss': loss}
            return {'acc': acc, 'loss': loss, 'log': log}



    
    def training_step(self, batch, batch_idx):
        return self._step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        
        return self._step("val", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self._step("test", batch, batch_idx)

    def _epoch_end(self, prefix, outputs):   
        if not outputs:
            return {}
        acc_mean = 0
        loss_mean = torch.mean(torch.tensor([out["loss"] for out in outputs]))
        if prefix == 'val':
          acc_mean = torch.mean(torch.tensor([out["acc"] for out in outputs], dtype=torch.float)) 
        
        log = {
            f'{prefix}_loss': loss_mean,
            f'{prefix}_acc': acc_mean,
        }
        
        result = {'progress_bar': log, 'log': log}

        return result

    def get_query_encoder(self):
        return self.query_encoder

    def get_doc_encoder(self):
        return self.doc_encoder
    
    def training_epoch_end(self, outputs):
        return self._epoch_end("train", outputs)

    def validation_epoch_end(self, outputs):
        return self._epoch_end("val", outputs)

    def test_epoch_end(self, outputs):
        return self._epoch_end("test", outputs)
