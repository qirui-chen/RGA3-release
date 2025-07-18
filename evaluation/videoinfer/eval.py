import numpy as np

import argparse
import json
import os
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from pycocoevalcap.bleu.bleu import Bleu as Bleuold
from pycocoevalcap.bleu.bleu_scorer import BleuScorer
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor # apt-get install default-jre
from pycocoevalcap.rouge.rouge import Rouge
from collections import Counter, defaultdict
from tqdm import tqdm


class Bleu(Bleuold):

    def compute_score(self, gts, res):

        assert (gts.keys() == res.keys())
        imgIds = gts.keys()

        bleu_scorer = BleuScorer(n=self._n)
        for id in imgIds:
            hypo = res[id]
            ref = gts[id]
            assert (type(hypo) is list)
            assert (len(hypo) == 1)
            assert (type(ref) is list)
            assert (len(ref) >= 1)
            bleu_scorer += (hypo[0], ref)
        score, scores = bleu_scorer.compute_score(option='closest', verbose=0)

        return score, scores


class SentenceTransformerSimilarity:
    
    def __init__(self, model_name='path/to/all-MiniLM-L6-v2', device='cuda'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def compute_score(self, gts, res):
        scores = []
        assert (gts.keys() == res.keys())
        imgIds = gts.keys()

        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            # Tokenize sentences
            encoded_hypo = self.tokenizer(hypo, padding=True, truncation=True, return_tensors='pt').to(self.device)
            encoded_ref = self.tokenizer(ref, padding=True, truncation=True, return_tensors='pt').to(self.device)

            # Compute token embeddings
            with torch.no_grad():
                hypo_output = self.model(**encoded_hypo)
                ref_output = self.model(**encoded_ref)

            # Perform pooling
            hypo_embedding = self.mean_pooling(hypo_output, encoded_hypo['attention_mask'])
            ref_embedding = self.mean_pooling(ref_output, encoded_ref['attention_mask'])

            # Normalize embeddings
            hypo_embedding = F.normalize(hypo_embedding, p=2, dim=1)
            ref_embedding = F.normalize(ref_embedding, p=2, dim=1)

            # Compute dot score
            score = torch.matmul(hypo_embedding, ref_embedding.transpose(0, 1))[0, 0].cpu().item()
            scores.append(score)

        score = np.mean(scores)

        return score, scores



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--pred_file', type=str, default="")
    parser.add_argument('--gt_file', type=str, default="")
    parser.add_argument('--results_file', type=str, default="")
    args = parser.parse_args()

        
    predictions = json.load(open(args.pred_file))
    labels = json.load(open(args.gt_file))


    bleu4_scorer = Bleu(4)
    meteor_scorer = Meteor()
    rouge_scorer = Rouge()
    cider_scorer = Cider()
    similarity_scorer = SentenceTransformerSimilarity()

    all_pred_answers, all_gt_answers, id2idx = {}, {}, {}
    
    cnt = 0
    for vid, vid_data in labels.items():
        for exp_id, exp_data in vid_data['expressions'].items():
            for qa_id, qa in exp_data['QA'].items():

                question = qa['Q']
                gt_answer = qa['A']
                pred_answer = predictions[vid][exp_id][qa_id]

                sample_id = f"{vid}_{exp_id}_{qa_id}"
                all_pred_answers[sample_id] = [pred_answer.replace("\n", " ")]
                all_gt_answers[sample_id] = [gt_answer]

                id2idx[sample_id] = cnt
                cnt += 1


    # Calculate BLEU scores
    bleu4_score, bleu4_scores = bleu4_scorer.compute_score(all_gt_answers, all_pred_answers)
    # Calculate METEOR scores
    meteor_score, meteor_scores = meteor_scorer.compute_score(all_gt_answers, all_pred_answers)
    # Calculate ROUGE scores, focusing on ROUGE-L
    rouge_l_score, rouge_l_scores = rouge_scorer.compute_score(all_gt_answers, all_pred_answers)
    # Calculate CIDER scores
    cider_score, cider_scores = cider_scorer.compute_score(all_gt_answers, all_pred_answers)
    # Calculate Similarity scores
    similarity_score, similarity_scores = similarity_scorer.compute_score(all_gt_answers, all_pred_answers)


    eval_results = {}
    for vid, vid_data in labels.items():
        for exp_id, exp_data in vid_data['expressions'].items():
            for qa_id, qa in exp_data['QA'].items():

                question = qa['Q']
                gt_answer = qa['A']
                pred_answer = predictions[vid][exp_id][qa_id]
                sample_id = f"{vid}_{exp_id}_{qa_id}"

                idx = id2idx[sample_id]

                if vid not in eval_results:
                    eval_results[vid] = {}
                if exp_id not in eval_results[vid]:
                    eval_results[vid][exp_id] = {}

                eval_results[vid][exp_id][qa_id] = {
                    "question": question,
                    "gt_answer": gt_answer,
                    "pred_answer": pred_answer,

                    "BLEU-4": bleu4_scores[3][idx],
                    "METEOR": meteor_scores[idx],
                    "ROUGE-L": rouge_l_scores[idx],
                    "CIDEr": cider_scores[idx],
                    "Similarity": similarity_scores[idx],
                }



    print("=" * 60)
    print(f"BLEU-4: {bleu4_score[3]*100:.1f}")
    print(f"METEOR: {meteor_score*100:.1f}")
    print(f"ROUGE-L: {rouge_l_score*100:.1f}")
    print(f"CIDEr: {cider_score*100:.1f}")
    print(f"Similarity: {similarity_score*100:.1f}")


    os.makedirs(os.path.dirname(args.results_file), exist_ok=True)
    with open(args.results_file, "w") as f:
        json.dump(eval_results, f, indent=2)