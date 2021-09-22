import time
import os
import torch
import random
import numpy as np
from scipy.optimize import linear_sum_assignment
import copy
import re
import string
import spacy


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def hungarian_matcher(predicted_spans, target_spans):
    """
    Args:
        predictions: prediction of one arg role type, list of [s,e]
        targets: target of one arg role type, list of [s,e]
    Return:
        (index_i, index_j) where index_i in prediction, index_j in target 
    """
    # L1 cost between spans
    cost_spans = torch.cdist(torch.FloatTensor(predicted_spans).unsqueeze(0), torch.FloatTensor(target_spans).unsqueeze(0), p=1)
    indices = linear_sum_assignment(cost_spans.squeeze(0)) 
    return [torch.as_tensor(indices[0], dtype=torch.int64), torch.as_tensor(indices[1], dtype=torch.int64)]

def _normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace. (Squad Style) """
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    s_normalized = white_space_fix(remove_articles(remove_punc(lower(s))))
    return s_normalized

def _tok_idx2word_idx(span, new_tok_index_to_old_tok_index, offset):
    span = list(span)
    span[0] = min(span[0], max(new_tok_index_to_old_tok_index.keys()))
    span[1] = max(span[1]-1, min(new_tok_index_to_old_tok_index.keys()))

    while span[0] not in new_tok_index_to_old_tok_index:
        span[0]+=1 
    span_s = new_tok_index_to_old_tok_index[span[0]]+offset
    while span[1] not in new_tok_index_to_old_tok_index:
        span[1]-=1 
    span_e = new_tok_index_to_old_tok_index[span[1]]+offset
    while span_e < span_s:
        span_e+=1
    return (span_s, span_e)

def _new_tok_id2old_tok_id(old_tok_to_new_tok_index):
    new_tok_index_to_old_tok_index = dict()
    for old_tok_id, (new_tok_id_s, new_tok_id_e) in enumerate(old_tok_to_new_tok_index):
        for j in range(new_tok_id_s, new_tok_id_e):
            new_tok_index_to_old_tok_index[j] = old_tok_id 
    return new_tok_index_to_old_tok_index


def eval_score_std_span(features, dset_type):
    # evaluate both text and standard span in annotated word tokens
    gt_num, pred_num, correct_num, correct_text = 0, 0, 0, 0 
    
    for feature in features:
        new_tok_index_to_old_tok_index = _new_tok_id2old_tok_id(feature.old_tok_to_new_tok_index)
        
        # NOTE deal with missing aligned ids
        if dset_type=='ace_eeqa':
            offset=0 # since eeqa annotation do not contain full text but partial text, although offset is provided
        else:
            offset = feature.event_trigger[2]

        for arg_role in feature.arg_list:
            pred_list=list(); gt_list = list()
            for span in feature.pred_dict[arg_role]:
                if span != (0,0):
                    pred_list.append(_tok_idx2word_idx(span, new_tok_index_to_old_tok_index, offset) )
            pred_list = list(set(pred_list))

            if arg_role in feature.gt_dict:
                for span in feature.gt_dict[arg_role]:
                    if span != (0,0):
                        gt_list.append(_tok_idx2word_idx(span, new_tok_index_to_old_tok_index, offset))

            pred_list_copy = copy.deepcopy(pred_list)
            gt_num += len(gt_list)
            pred_num += len(pred_list)
            
            if len(gt_list) != 0:
                correct_list = []
                for gt_span in gt_list:
                    if gt_span in pred_list:
                        correct_list.append(pred_list.pop(pred_list.index(gt_span)))
                correct_num += len(correct_list)

            feature.pred_dict[arg_role] = pred_list_copy
            feature.gt_dict[arg_role] = gt_list

            full_text = feature.full_text
            gt_texts = [_normalize_answer(" ".join(full_text[gt_span[0]: gt_span[1]+1])) for gt_span in gt_list]
            #gt_texts = [" ".join(full_text[gt_span[0]: gt_span[1]+1]).lower().strip() for gt_span in gt_list]
            if len(gt_texts) != 0:
                pred_texts = [_normalize_answer(" ".join(full_text[pred_span[0]: pred_span[1]+1])) for pred_span in pred_list_copy]
                #pred_texts = [" ".join(full_text[pred_span[0]: pred_span[1]+1]).lower().strip() for pred_span in pred_list_copy]
                correct_list = []
                for gt_text in gt_texts:
                    if gt_text in pred_texts:
                        correct_list.append(pred_texts.pop(pred_texts.index(gt_text)))
                correct_text += len(correct_list)
            
    recall = correct_num/gt_num if gt_num!=0 else .0
    precision = correct_num/pred_num if pred_num!=0 else .0
    f1 = 2*recall*precision/(recall+precision) if (recall+precision)>1e-4 else .0

    recall_text = correct_text/gt_num if gt_num!=0 else .0
    precision_text = correct_text/pred_num if pred_num!=0 else .0
    f1_text = 2*recall_text*precision_text/(recall_text+precision_text) if (recall_text+precision_text)>1e-4 else .0

    return [recall, precision, f1, gt_num, pred_num, correct_num], [recall_text, precision_text, f1_text, gt_num, pred_num, correct_text]

def show_results(features, output_file, metainfo):
    """ paie std show resuults """
    with open(output_file, 'w', encoding='utf-8') as f:
        for k,v in metainfo.items():
            f.write(f"{k}: {v}\n")

        for feature in features:
            example_id = feature.example_id
            
            sent = feature.enc_text
            f.write("-------------------------------------------------------------------------------------\n")
            f.write("Sent: {}\n".format(sent))
            f.write("Event type: {}\t\t\tTrigger word: {}\n".format(feature.event_type, feature.event_trigger))
            f.write("Example ID {}\n".format(example_id))
            full_text = feature.full_text
            for arg_role in feature.arg_list:
                
                pred_list = feature.pred_dict[arg_role] 
                gt_list = feature.gt_dict[arg_role]
                if len(pred_list)==0 and len(gt_list)==0:
                    continue
                
                if len(gt_list) == 0 and len(pred_list) > 0:
                    gt_list = [(-1,-1)] * len(pred_list)
                
                if len(gt_list) > 0 and len(pred_list) == 0:
                    pred_list = [(-1,-1)] * len(gt_list)

                gt_idxs, pred_idxs = hungarian_matcher( gt_list, pred_list)

                for pred_idx, gt_idx in zip(pred_idxs, gt_idxs):
                    if gt_list[gt_idx] == (-1,-1) and pred_list[pred_idx] == (-1,-1):
                        continue
                    else:
                        pred_text = " ".join(full_text[pred_list[pred_idx][0]: pred_list[pred_idx][1]+1]) if pred_list[pred_idx]!=(-1,-1) else "__ No answer __"
                        gt_text = " ".join(full_text[gt_list[gt_idx][0]: gt_list[gt_idx][1]+1]) if gt_list[gt_idx]!=(-1,-1) else "__ No answer __"
                    
                    if gt_list[gt_idx] == pred_list[pred_idx]:
                        f.write("Arg {} matched: Pred: {} ({},{})\tGt: {} ({},{})\n".format(arg_role, pred_text, pred_list[pred_idx][0], pred_list[pred_idx][1], gt_text, gt_list[gt_idx][0], gt_list[gt_idx][1]))
                    else:
                        f.write("Arg {} dismatched: Pred: {} ({},{})\tGt: {} ({},{})\n".format(arg_role, pred_text, pred_list[pred_idx][0], pred_list[pred_idx][1], gt_text, gt_list[gt_idx][0], gt_list[gt_idx][1]))
                
                if len(gt_idxs) < len(gt_list): # prediction  __no answer__
                    for idx in range(len(gt_list)):
                        if idx not in gt_idxs:
                            gt_text = gt_text = " ".join(full_text[gt_list[idx][0]: gt_list[idx][1]+1])
                            f.write("Arg {} dismatched: Pred: {} ({},{})\tGt: {} ({},{})\n".format(arg_role, "__ No answer __", -1, -1, gt_text, gt_list[idx][0], gt_list[idx][1])) 

                if len(pred_idxs) < len(pred_list): # ground truth  __no answer__
                    for idx in range(len(pred_list)):
                        if idx not in pred_idxs:
                            pred_text = " ".join(full_text[pred_list[idx][0]: pred_list[idx][1]+1])
                            f.write("Arg {} dismatched: Pred: {} ({},{})\tGt: {} ({},{})\n".format(arg_role, pred_text, pred_list[idx][0], pred_list[idx][1], "__ No answer __", -1, -1))


def get_best_index(feature, start_logit, end_logit, max_span_length, max_span_num):

    th = start_logit[0] + end_logit[0]
    answer_span_list = []
    context_length = len(feature.old_tok_to_new_tok_index)
    cal_time, index_time, cmp_time = 0, 0, 0

    for start in range(context_length):
        for end in range(start+1, min(context_length, start+max_span_length+1)):
            t1 = time.time()
            start_index = feature.old_tok_to_new_tok_index[start][0] # use start token idx
            end_index = feature.old_tok_to_new_tok_index[end-1][1] 
            t2 = time.time()
            index_time += (t2-t1)

            t1 = time.time()
            score = start_logit[start_index] + end_logit[end_index]
            answer_span = (start_index, end_index, score)
            t2 = time.time()
            cal_time += (t2-t1)

            t1 = time.time()
            if score > th:
                answer_span_list.append(answer_span)
            t2 = time.time()
            cmp_time += (t2-t1)
    
    if not answer_span_list:
        answer_span_list.append((0, 0, th))
    return filter_spans(answer_span_list, max_span_num), index_time, cal_time, cmp_time

def filter_spans(candidate_span_list, max_span_num):
    candidate_span_list = sorted(candidate_span_list, key=lambda x:x[2], reverse=True)
    candidate_span_list = [(candidate_span[0], candidate_span[1]) for candidate_span in candidate_span_list]

    def is_intersect(span_1, span_2):
        return False if min(span_1[1], span_2[1]) < max(span_1[0], span_2[0]) else True

    if len(candidate_span_list) == 1:
        answer_span_list = candidate_span_list
    else:
        answer_span_list = []
        while candidate_span_list and len(answer_span_list)<max_span_num:
            selected_span = candidate_span_list[0]
            answer_span_list.append(selected_span)
            candidate_span_list = candidate_span_list[1:]  

            candidate_span_list = [candidate_span for candidate_span in candidate_span_list if not is_intersect(candidate_span, selected_span)]
    return answer_span_list

def check_tensor(tensor, var_name):
    print("******Check*****")
    print("tensor_name: {}".format(var_name))
    print("shape: {}".format(tensor.size()))
    if len(tensor.size())==1 or tensor.size(0)<=3:
        print("value: {}".format(tensor))
    else:
        print("part value: {}".format(tensor[0,:]))
    print("require_grads: {}".format(tensor.requires_grad))
    print("tensor_type: {}".format(tensor.dtype))


# FULL Evaluation Metric
from spacy.tokens import Doc
class WhitespaceTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(" ")
        return Doc(self.vocab, words=words)

def find_head(arg_start, arg_end, doc):
    arg_end -= 1

    cur_i = arg_start
    while doc[cur_i].head.i >= arg_start and doc[cur_i].head.i <=arg_end:
        if doc[cur_i].head.i == cur_i:
            # self is the head 
            break 
        else:
            cur_i = doc[cur_i].head.i
        
    arg_head = cur_i
    head_text = doc[arg_head]
    
    return head_text


def eval_score_std_span_full_metrics(features, dset_type):
    nlp = spacy.load('en_core_web_sm')
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

    # evaluate both text and standard span in annotated word tokens
    gt_num, pred_num = 0, 0
    correct_num, correct_text_num = 0, 0 
    correct_head_num, correct_identify_num = 0, 0
    
    last_full_text = None
    for feature in features:
        new_tok_index_to_old_tok_index = _new_tok_id2old_tok_id(feature.old_tok_to_new_tok_index)
        
        # NOTE deal with missing aligned ids
        if dset_type=='eeqa':
            offset=0 # since eeqa annotation do not contain full text but partial text, although offset is provided
        else:
            offset = feature.event_trigger[2]

        all_pred_list = list(); all_gt_list = list()
        for arg_role in feature.arg_list:
            pred_list = list(); gt_list = list()
            for span in feature.pred_dict[arg_role]:
                if span != (0,0):
                    pred_list.append(_tok_idx2word_idx(span, new_tok_index_to_old_tok_index, offset) )
            pred_list = list(set(pred_list))

            if arg_role in feature.gt_dict:
                for span in feature.gt_dict[arg_role]:
                    if span != (0,0):
                        gt_list.append(_tok_idx2word_idx(span, new_tok_index_to_old_tok_index, offset))

            pred_list_copy = copy.deepcopy(pred_list)
            gt_num += len(gt_list)
            pred_num += len(pred_list)
            
            if len(gt_list) != 0:
                correct_list = []
                for gt_span in gt_list:
                    if gt_span in pred_list:
                        correct_list.append(pred_list.pop(pred_list.index(gt_span)))
                correct_num += len(correct_list)

            feature.pred_dict[arg_role] = pred_list_copy
            feature.gt_dict[arg_role] = gt_list

            full_text = feature.full_text
            if full_text!=last_full_text:
                doc = nlp(" ".join(full_text))
                last_full_text = full_text
            gt_texts = [_normalize_answer(" ".join(full_text[gt_span[0]: gt_span[1]+1])) for gt_span in gt_list]
            gt_head_texts = [str(find_head(gt_span[0], gt_span[1]+1, doc)) for gt_span in gt_list]

            pred_texts = [_normalize_answer(" ".join(full_text[pred_span[0]: pred_span[1]+1])) for pred_span in pred_list_copy]
            pred_head_texts = [str(find_head(pred_span[0], pred_span[1]+1, doc)) for pred_span in pred_list_copy]
            pred_texts_copy = copy.deepcopy(pred_texts)
            if len(gt_texts) != 0:
                correct_list = []
                for gt_text in gt_texts:
                    if gt_text in pred_texts:
                        correct_list.append(pred_texts.pop(pred_texts.index(gt_text)))
                correct_text_num += len(correct_list)

                for gt_head in gt_head_texts:
                    if gt_head in pred_head_texts:
                        correct_head_num += 1
            # use span
            all_pred_list.extend(pred_list_copy)
            all_gt_list.extend(gt_list)

        for gt_span in all_gt_list:
            if gt_span in all_pred_list:
                all_pred_list.pop(all_pred_list.index(gt_span))
                correct_identify_num+=1
        
    recall = correct_num/gt_num if gt_num!=0 else .0
    precision = correct_num/pred_num if pred_num!=0 else .0
    f1 = 2*recall*precision/(recall+precision) if (recall+precision)>1e-4 else .0

    recall_text = correct_text_num/gt_num if gt_num!=0 else .0
    precision_text = correct_text_num/pred_num if pred_num!=0 else .0
    f1_text = 2*recall_text*precision_text/(recall_text+precision_text) if (recall_text+precision_text)>1e-4 else .0

    recall_identify = correct_identify_num/gt_num if gt_num!=0 else .0
    precision_identify = correct_identify_num/pred_num if pred_num!=0 else .0
    f1_identify = 2*recall_identify*precision_identify/(recall_identify+precision_identify) if (recall_identify+precision_identify)>1e-4 else .0

    recall_head = correct_head_num/gt_num if gt_num!=0 else .0
    precision_head = correct_head_num/pred_num if pred_num!=0 else .0
    f1_head = 2*recall_head*precision_head/(recall_head+precision_head) if (recall_head+precision_head)>1e-4 else .0

    return [recall, precision, f1, gt_num, pred_num, correct_num], [recall_text, precision_text, f1_text, gt_num, pred_num, correct_text_num], \
        [recall_identify, precision_identify, f1_identify, gt_num, pred_num, correct_identify_num], [recall_head, precision_head, f1_head, gt_num, pred_num, correct_head_num]

