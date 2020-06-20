from dataset import *
from helpers import *
from modules import *
import torch
import torch.nn as nn
import pytorch_lightning as pl

class TwoTower(pl.LightningModule):
    """Two tower model for document retrieval using query and document."""
    def __init__(self,
                 learning_rate=1e-3,
                 batch_size=32,
                 dim=128,
                 epochs = None,
                 max_length=32,
                 shuffle=True,
                 queries_train_path = queries_train_path,
                 queries_dev_path = queries_dev_path,
                 docs_path = doc2queries_path,
                 triples_path = triples_path,
                 top1000_dev_path = top1000_path,
                 qrels_dev_path = qrels_dev_path,
                 batch_size_val = 5
                 ):
        super(TwoTower, self).__init__()

        #Data
        self.queries_train = load_queries(queries_train_path)
        self.queries_dev = load_queries(queries_dev_path)
        self.docs_queries = load_doc2query(docs_path)
        self.triples = load_triple(triples_path, 10_000)
        self.queries_top1000 = load_top1000_dev(top1000_path)
        qrels = load_qrels(qrels_dev_path)
        self.qrels = {int(qid):docids for qid,docids in qrels.items()}

        self.embedding_dim = dim
        self.vectors = [] 

        # Configuration
        self.learning_rate = learning_rate
        self.batch_size=batch_size
        self.batch_size_val = batch_size_val
        self.max_length = max_length
        self.loss = CosineSimilarityLoss()
        self.similarity = nn.CosineSimilarity(dim=-1, eps=1e-08)
        self.optimizer = torch.optim.Adam
        self.opt_params = {'lr': learning_rate, 'eps': 1e-08, 'betas': (0.9, 0.999)}
        self.warmup_steps = 500
        self.training_steps = (epochs*len(self.triples))/batch_size if epochs != None else None
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)


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
        
        self.train_data = MyDataset(
                triples = self.triples,
                queries = self.queries_train,
                docs = self.docs_queries,
                max_length = self.max_length,
                tokenizer=self.tokenizer)
        
        self.valid_data = ValDataset(
                queries = self.queries_dev,
                docs = self.docs_queries,
                tokenizer = self.tokenizer,
                max_length = self.max_length,
                queries_1000 = self. queries_top1000)
    
    
        

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
          
          loss, sim_pos, sim_neg = self.loss(query_embedding, p_doc_embedding, n_doc_embedding) 
          
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
          
          _, indices = torch.sort(scores)

          doc_ids_sorted = []
          for i, doc_id in enumerate(doc_ids):
            doc_ids_sorted.append(doc_id[indices[i]].tolist())
          
          qrel = {}
          for i, doc_ids in enumerate(doc_ids_sorted):
            qrel[int(qid[i])] = doc_ids

          return qrel

    def _step(self, prefix, batch, batch_nb):

        if prefix == 'train':
            loss, _, _ = self(batch)
            log = {f'{prefix}_loss': loss}
            return {'loss': loss, 'log': log,}

        elif(prefix=='val'):
            qrel_pred = self(batch)
            mrr_dict = msmarco_eval.compute_metrics(self.qrels,qrel_pred)

            mrr = mrr_dict['MRR @10']
            n_queries = mrr_dict['QueriesRanked']

            log = {f'{prefix}_mrr': mrr}
            return {'mrr': mrr, 'log': log}

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
        loss_mean = 0

        if prefix == 'train':
          loss_mean = torch.mean(torch.tensor([out["loss"] for out in outputs]))
          log = {
            f'{prefix}_loss': loss_mean
            } 
        if prefix == 'val':
          mrr_mean = torch.mean(torch.tensor([out["mrr"] for out in outputs], dtype=torch.float)) 
          log = {
            f'{prefix}_mrr': mrr_mean,
            } 
        


        return {'progress_bar': log, 'log': log}

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

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=int, default=1e-3)
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--dim', type=int, default=64)
        parser.add_argument('--epochs', type=int, default=None)
        parser.add_argument('--max_lenght', type=int, default=32)
        parser.add_argument('--shuffle', type=bool, default=True)
        parser.add_argument('--queries-path', type=str)
        parser.add_argument('--docs-path', type=str)
        parser.add_argument('--triples-path', type=str)
        return parser
