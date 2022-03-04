def _new_tok_id2old_tok_id(old_tok_to_new_tok_index):
    new_tok_index_to_old_tok_index = dict()
    for old_tok_id, (new_tok_id_s, new_tok_id_e) in enumerate(old_tok_to_new_tok_index):
        for j in range(new_tok_id_s, new_tok_id_e):
            new_tok_index_to_old_tok_index[j] = old_tok_id 
    return new_tok_index_to_old_tok_index


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


def eval_std_cls_f1_score(features, dset_type):
    gt_num, pred_num, correct_num = 0, 0, 0

    for feature in features:
        new_tok_index_to_old_tok_index = _new_tok_id2old_tok_id(feature.old_tok_to_new_tok_index)
        
        # NOTE deal with missing aligned ids
        if dset_type=='ace_eeqa':
            offset=0 # since eeqa annotation do not contain full text but partial text, although offset is provided
        else:
            offset = feature.event_trigger[2]

        for arg_role in feature.arg_list:
            pred_list = list()
            gt_list = list()
            for span in feature.pred_dict[arg_role]:
                if span != (0,0):
                    pred_list.append(_tok_idx2word_idx(span, new_tok_index_to_old_tok_index, offset) )
            pred_list = list(set(pred_list))

            if arg_role in feature.gt_dict:
                for span in feature.gt_dict[arg_role]:
                    if span != (0,0):
                        gt_list.append(_tok_idx2word_idx(span, new_tok_index_to_old_tok_index, offset))

            gt_num += len(gt_list)
            pred_num += len(pred_list)
            for gt_span in gt_list:
                if gt_span in pred_list:
                    correct_num += 1
            
    recall = correct_num/gt_num if gt_num!=0 else .0
    precision = correct_num/pred_num if pred_num!=0 else .0
    f1 = 2*recall*precision/(recall+precision) if (recall+precision)>1e-4 else .0

    return recall, precision, f1, gt_num, pred_num, correct_num





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
        if dset_type=='ace_eeqa':
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