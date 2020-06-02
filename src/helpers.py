import collections
import functools
import traceback

def gpu_mem_restore(func):
    "Reclaim GPU RAM if CUDA out of memory happened, or execution was interrupted"

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            type, val, tb = sys.exc_info()
            traceback.clear_frames(tb)
            raise type(val).with_traceback(tb) from None

    return wrapper

def recall_at_k(qrel, ans, k=100):
    """Compute the recall@k given qrel, search outputs in the form of a dict and k."""
    n = 0
    total = 0
    for qid, v in zip(ans.keys(), ans.values()):
        if len(qrel[qid]):
            acc = sum([1 for e in v[:k] if e in qrel[qid]])
            recall = acc/len(qrel[qid])
            total += recall
            n += 1
    return total/n


# From https://github.com/nyu-dl/dl4ir-doc2query/blob/master/convert_msmarco_to_opennmt.py
#
# BSD 3-Clause License
#
# Copyright (c) 2019, 
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

def load_qrels(path):
  """Loads qrels into a dict of key: query id, value: set of relevant doc ids."""
  qrels = collections.defaultdict(set)
  with open(path) as f:
    for i, line in enumerate(f):
      query_id, _, doc_id, relevance = line.rstrip().split('\t')
      if int(relevance) >= 1:
        qrels[query_id].add(doc_id)
      if i % 100000 == 0:
        print('Loading qrels {}'.format(i))
  return qrels


def load_queries(path):
  """Loads queries into a dict of key: query id, value: query text."""
  queries = {}
  with open(path) as f:
    for i, line in enumerate(f):
      query_id, query = line.rstrip().split('\t')
      queries[query_id] = query
      if i % 100000 == 0:
        print('Loading queries {}'.format(i))
  return queries


def load_collection(path):
  """Loads tsv collection into a dict of key: doc id, value: doc text."""
  collection = {}
  with open(path) as f:
    for i, line in enumerate(f):
      doc_id, doc_text = line.rstrip().split('\t')
      collection[doc_id] = doc_text.replace('\n', ' ')
      if i % 1000000 == 0:
        print('Loading collection, doc {}'.format(i))

  return collection

#----------------------

def load_doc2query(path):
  """Loads txt queries from doc2query into a dict of key: doc id, value: doc query."""
  collection = {}
  with open(path) as f:
    for i, line in enumerate(f):
      doc_text = line.rstrip()
      collection[str(i)] = doc_text.replace('\n', ' ')
      if i % 1000000 == 0:
        print('Loading doc2query, doc {}'.format(i))

  return collection
