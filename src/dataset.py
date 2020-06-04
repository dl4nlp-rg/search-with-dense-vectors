from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, triples, queries, docs, tokenizer, max_length):
        self.triples = triples
        self.queries = queries
        self.docs = docs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        qid, pid, nid = self.triples[idx]

        query = self.queries[qid]
        doc_pos = self.docs[pid]
        doc_neg = self.docs[nid]

        q_tok, q_mask, q_type = self.tokenize(query)
        p_tok, p_mask, p_type = self.tokenize(doc_pos)
        n_tok, n_mask, n_type = self.tokenize(doc_neg)

        return  (q_tok, q_mask, q_type,
                p_tok, p_mask, p_type,
                n_tok, n_mask, n_type,
                query, doc_pos, doc_neg)

    def tokenize(self, text):
        tokens = self.tokenizer.encode_plus(text=text, max_length=self.max_length,
                                       pad_to_max_length=True, add_special_tokens = True,
                                       return_tensors='pt')
        return tokens["input_ids"], tokens['attention_mask'], tokens['token_type_ids']
