from .dataset import *
from .helpers import *
from .modules import *
from . import msmarco_eval
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import BertTokenizer
from argparse import ArgumentParser

class TwoTower(pl.LightningModule):
    """Two tower model for document retrieval using query and document."""
    def __init__(self,
                 queries_train_path,
                 queries_dev_path,
                 docs_path,
                 triples_path,
                 top1000_path,
                 qrels_dev_path,
                 learning_rate=1e-3,
                 batch_size=32,
                 dim=128,
                 epochs = None,
                 warmup_steps = 50,
                 max_length=32,
                 shuffle=True,
                 batch_size_val = 5,
                 pretrained_model='bert-base-uncased',
                 max_val=100,
                 max_train=100_000,
                 **kwargs
                 ):
        super(TwoTower, self).__init__()

        # Data
        self.queries_train_path = queries_train_path
        self.queries_dev_path = queries_dev_path
        self.docs_path = docs_path
        self.triples_path = triples_path
        self.top1000_path = top1000_path
        self.qrels_dev_path = qrels_dev_path
        self.max_train = max_train
        self.max_val = max_val

        # Configuration
        self.learning_rate = learning_rate
        self.batch_size=batch_size
        self.batch_size_val = batch_size_val
        self.max_length = max_length
        self.warmup_steps = warmup_steps
        self.shuffle=shuffle
        self.pin_mem = True
        self.n_workers = 0
        self.embedding_dim = dim
        self.pretrained_model = pretrained_model
        self.epochs = epochs

    def setup(self, step):
        # Optimizer
        self.optimizer = torch.optim.Adam
        self.opt_params = {'lr': self.learning_rate, 'eps': 1e-08, 'betas': (0.9, 0.999)}

        # Models
        self.query_encoder = Encoder(self.embedding_dim)
        self.doc_encoder = Encoder(self.embedding_dim)

        # Loss
        self.loss_fn = CosineSimilarityLoss(k=1)
        self.similarity = nn.CosineSimilarity(dim=1, eps=1e-08)

        # Steps
        self.training_steps = (self.epochs*self.len_triples)/self.batch_size if self.epochs != None else None


    def prepare_data(self):
        # Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_model)

        # Download data
        queries_train = load_queries(self.queries_train_path)
        queries_dev = load_queries(self.queries_dev_path)
        docs_queries = load_doc2query(self.docs_path)
        triples = load_triple(self.triples_path, self.max_train)
        queries_top1000 = load_top1000_dev(self.top1000_path, self.max_val)
        qrels = load_qrels(self.qrels_dev_path)
        self.qrels = {int(qid): [int(e) for e in docids] for qid,docids in qrels.items()}
        self.len_triples = len(triples)

        # Train dataset
        self.train_data = MyDataset(
                triples = triples,
                queries = queries_train,
                docs = docs_queries,
                max_length = self.max_length,
                tokenizer=self.tokenizer)

        # Val dataset
        self.valid_data = ValDataset(
                queries = queries_dev,
                docs = docs_queries,
                tokenizer = self.tokenizer,
                max_length = self.max_length,
                queries_1000 = queries_top1000)

    @gpu_mem_restore
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size,
                          shuffle=self.shuffle, num_workers=self.n_workers,
                          pin_memory=self.pin_mem)

    @gpu_mem_restore
    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size=self.batch_size_val,
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
            if self.training_steps != None:
                lr_scale = max(0.0,
                    float(self.training_steps - self.trainer.global_step) / float(max(1,
                                                    self.training_steps - self.warmup_steps)))
                for pg in optimizer.param_groups:
                    pg['lr'] = lr_scale * self.learning_rate

        # update params
        optimizer.step()
        optimizer.zero_grad()
        
    def forward(self,batch):
        if self.training:
          q_tok, q_mask, q_type, p_tok, p_mask, p_type, n_tok, n_mask, n_type, query, doc_pos, doc_neg = batch

          query_embedding = self.query_encoder(q_tok.squeeze(-2),q_mask.squeeze(-2),q_type.squeeze(-2))
          p_doc_embedding = self.doc_encoder(p_tok.squeeze(-2),p_mask.squeeze(-2),p_type.squeeze(-2))
          n_doc_embedding = self.doc_encoder(n_tok.squeeze(-2),n_mask.squeeze(-2),n_type.squeeze(-2))

          loss, sim_pos, sim_neg = self.loss_fn(query_embedding, p_doc_embedding, n_doc_embedding)

          return loss.mean(), sim_pos, sim_neg

        else:
          qrel = {}
          q_tok, q_mask, q_type, docs_tok, docs_mask, docs_type, qid, doc_ids = batch

          #Corrige as dimensões dos elementos.
          docs_tok = docs_tok.view((-1,self.max_length))
          docs_mask = docs_mask.view((-1,self.max_length))
          docs_type = docs_type.view((-1,self.max_length))
          q_tok = q_tok.squeeze(-2)
          q_mask = q_mask.squeeze(-2)
          q_type = q_type.squeeze(-2)
          doc_ids = torch.tensor(correct_docids(doc_ids)) #Corrige a ordem dos doc_ids

          query_embedding = self.query_encoder(q_tok,q_mask,q_type)

          doc_embedding = self.doc_encoder(docs_tok,docs_mask,docs_type)
          doc_embedding = doc_embedding.view((self.batch_size_val,-1,self.embedding_dim))

          scores = self.similarity(query_embedding.unsqueeze(1), doc_embedding)

          _, indices = torch.sort(scores, descending=True)

          doc_ids_sorted = []
          for i, doc_id in enumerate(doc_ids):
            doc_ids_sorted.append(doc_id[indices[i]].tolist())

          qrel = {}
          for i, doc_ids in enumerate(doc_ids_sorted):
            qrel[int(qid[i])] = doc_ids

          return qrel

    def training_step(self, batch, batch_idx):
        loss, _, _ = self(batch)
        logs = {'loss': loss}
        progress = {'train_loss': loss}
        return {'loss': loss, 'log': logs, 'progress_bar': progress}

    def validation_step(self, batch, batch_idx):
        qrel_pred = self(batch)
        mrr_dict = msmarco_eval.compute_metrics(self.qrels,qrel_pred)
        mrr = mrr_dict['MRR @10']
        return {'mrr': mrr}

    def validation_epoch_end(self, outputs):
        mrr_mean = torch.mean(torch.tensor([out["mrr"] for out in outputs], dtype=torch.float))
        log = {'mrr': mrr_mean,}
        return {'progress_bar': log, 'log': log}

    def get_query_encoder(self):
        return self.query_encoder

    def get_doc_encoder(self):
        return self.doc_encoder

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--dim', type=int, default=128)
        parser.add_argument('--epochs', type=int, default=None)
        parser.add_argument('--warmup_steps', type=int, default=50)
        parser.add_argument('--batch_size_val', type=int, default=5)
        parser.add_argument('--max_lenght', type=int, default=32)
        parser.add_argument('--shuffle', type=bool, default=True)
        parser.add_argument('--queries_train_path', type=str, default='queries.train.tsv')
        parser.add_argument('--queries_dev_path', type=str, default='queries.dev.small.tsv')
        parser.add_argument('--docs_path', type=str, default='collection.tsv')
        parser.add_argument('--triples_path', type=str, default='qidpidtriples.train.full.tsv')
        parser.add_argument('--top1000_path', type=str, default='top1000.dev')
        parser.add_argument('--qrels_dev_path', type=str, default='qrels.dev.small.tsv')
        parser.add_argument('--pretrained_model', type=str, default='bert-base-uncased')
        parser.add_argument('--max_train', type=int, default=100000)
        parser.add_argument('--max_val', type=int, default=100)
        return parser
