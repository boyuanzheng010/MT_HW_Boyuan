import optparse
import sys
from typing import Dict, List, Tuple, Union
import numpy as np
import torch
from transformers import BertModel, BertTokenizer, AutoConfig, AutoModel, AutoTokenizer
import os

class EmbeddingLoader(object):
    def __init__(self, model: str = "bert-base-multilingual-cased", device=torch.device('cpu'), layer: int = 8):
        TR_Models = {
            'bert-base-multilingual-cased': (BertModel, BertTokenizer),
            'bert-base-multilingual-uncased': (BertModel, BertTokenizer),
        }

        self.model = model
        self.device = device
        self.layer = layer
        self.emb_model = None
        self.tokenizer = None

        if model in TR_Models:
            model_class, tokenizer_class = TR_Models[model]
            self.emb_model = model_class.from_pretrained(model, output_hidden_states=True)
            self.emb_model.eval()
            self.emb_model.to(self.device)
            self.tokenizer = tokenizer_class.from_pretrained(model)
        else:
            if os.path.isdir(model):
                # try to load model with auto-classes
                config = AutoConfig.from_pretrained(model, output_hidden_states=True)
                self.emb_model = AutoModel.from_pretrained(model, config=config)
                self.emb_model.eval()
                self.emb_model.to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(model)
            else:
                raise ValueError("The model '{}' is not recognised!".format(model))

    def get_embed_list(self, sent_batch: List[List[str]]) -> torch.Tensor:
        if self.emb_model is not None:
            with torch.no_grad():
                if not isinstance(sent_batch[0], str):
                    inputs = self.tokenizer(sent_batch, is_split_into_words=True, padding=True, truncation=True,
                                            return_tensors="pt")
                else:
                    inputs = self.tokenizer(sent_batch, is_split_into_words=False, padding=True, truncation=True,
                                            return_tensors="pt")
                outputs = self.emb_model(**inputs.to(self.device))[2][self.layer]
                return outputs[:, 1:-1, :]
        else:
            return None

class SentenceAligner(object):
    def __init__(self, model: str = "bert", token_type: str = "bpe", distortion: float = 0.0, device: str = "cpu",
                 layer: int = 8):
        model_names = {
            "bert": "bert-base-multilingual-cased",
        }

        self.model = model
        if model in model_names:
            self.model = model_names[model]
        self.token_type = token_type
        self.distortion = distortion
        self.matching_methods = ["inter"]
        self.device = torch.device(device)

        self.embed_loader = EmbeddingLoader(model=self.model, device=self.device, layer=layer)

    @staticmethod
    def get_similarity(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        results = []
        for x in X:
            temp_list = []
            for y in Y:
                dot_product = np.dot(x.T, y)
                x_abs = np.sqrt(np.dot(x.T, x))
                y_abs = np.sqrt(np.dot(y.T, y))
                similarity = dot_product / (x_abs * y_abs)
                temp_list.append(similarity)
            results.append(temp_list)
        results = np.array(results)
        return results

    @staticmethod
    def average_embeds_over_words(bpe_vectors: np.ndarray, word_tokens_pair: List[List[str]]) -> List[np.array]:
        w2b_map = []
        cnt = 0
        w2b_map.append([])
        for wlist in word_tokens_pair[0]:
            w2b_map[0].append([])
            for x in wlist:
                w2b_map[0][-1].append(cnt)
                cnt += 1
        cnt = 0
        w2b_map.append([])
        for wlist in word_tokens_pair[1]:
            w2b_map[1].append([])
            for x in wlist:
                w2b_map[1][-1].append(cnt)
                cnt += 1

        new_vectors = []
        for l_id in range(2):
            w_vector = []
            for word_set in w2b_map[l_id]:
                w_vector.append(bpe_vectors[l_id][word_set].mean(0))
            new_vectors.append(np.array(w_vector))
        return new_vectors

    @staticmethod
    def get_alignment_matrix(sim_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        m, n = sim_matrix.shape
        forward = np.eye(n)[sim_matrix.argmax(axis=1)]  # m x n
        backward = np.eye(m)[sim_matrix.argmax(axis=0)]  # n x m
        return forward, backward.transpose()

    @staticmethod
    def apply_distortion(sim_matrix: np.ndarray, ratio: float = 0.5) -> np.ndarray:
        shape = sim_matrix.shape
        if (shape[0] < 2 or shape[1] < 2) or ratio == 0.0:
            return sim_matrix

        pos_x = np.array([[y / float(shape[1] - 1) for y in range(shape[1])] for x in range(shape[0])])
        pos_y = np.array([[x / float(shape[0] - 1) for x in range(shape[0])] for y in range(shape[1])])
        distortion_mask = 1.0 - ((pos_x - np.transpose(pos_y)) ** 2) * ratio

        return np.multiply(sim_matrix, distortion_mask)

    def get_word_aligns(self, src_sent: Union[str, List[str]], trg_sent: Union[str, List[str]]) -> Dict[str, List]:
        if isinstance(src_sent, str):
            src_sent = src_sent.split()
        if isinstance(trg_sent, str):
            trg_sent = trg_sent.split()
        l1_tokens = [self.embed_loader.tokenizer.tokenize(word) for word in src_sent]
        l2_tokens = [self.embed_loader.tokenizer.tokenize(word) for word in trg_sent]
        bpe_lists = [[bpe for w in sent for bpe in w] for sent in [l1_tokens, l2_tokens]]

        if self.token_type == "bpe":
            l1_b2w_map = []
            for i, wlist in enumerate(l1_tokens):
                l1_b2w_map += [i for x in wlist]
            l2_b2w_map = []
            for i, wlist in enumerate(l2_tokens):
                l2_b2w_map += [i for x in wlist]

        vectors = self.embed_loader.get_embed_list([src_sent, trg_sent]).cpu().detach().numpy()
        vectors = [vectors[i, :len(bpe_lists[i])] for i in [0, 1]]

        if self.token_type == "word":
            vectors = self.average_embeds_over_words(vectors, [l1_tokens, l2_tokens])

        all_mats = {}
        sim = self.get_similarity(vectors[0], vectors[1])
        sim = self.apply_distortion(sim, self.distortion)

        all_mats["fwd"], all_mats["rev"] = self.get_alignment_matrix(sim)
        all_mats["inter"] = all_mats["fwd"] * all_mats["rev"]

        aligns = {'inter': set()}
        for i in range(len(vectors[0])):
            for j in range(len(vectors[1])):
                for ext in self.matching_methods:
                    if all_mats[ext][i, j] > 0:
                        if self.token_type == "bpe":
                            aligns[ext].add((l1_b2w_map[i], l2_b2w_map[j]))
                        else:
                            aligns[ext].add((i, j))
        for ext in aligns:
            aligns[ext] = sorted(aligns[ext])
        return aligns


optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="train", default="split_data/fold_1/hansards",
                     help="Data filename prefix (default=data)")
optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float",
                     help="Threshold for aligning with Dice's coefficient (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=100000000000, type="int",
                     help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)

sys.stderr.write("Loading mBert-Aligner....")

model = SentenceAligner(model="bert")

with open('bert_inter.a', 'w', encoding='utf-8') as out_file:
    for (f, e) in zip(open(f_data, encoding='utf-8'), open(e_data, encoding='utf-8')):
        result = model.get_word_aligns(f, e)
        for (i, j) in result['inter']:
            out_file.write("%i-%i " % (i, j))
        out_file.write("\n")
