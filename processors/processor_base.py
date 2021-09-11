import os
import csv
import json
import jsonlines
import torch
import pickle

from itertools import chain
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import copy                             

import logging
logger = logging.getLogger(__name__)

class ACE_event:
    def __init__(self, doc_id, sent_id, sent, event_type, event_trigger, event_args, full_text):
        self.doc_id = doc_id
        self.sent_id = sent_id
        self.sent = sent
        self.type = event_type
        self.trigger = event_trigger
        self.args = event_args
        
        self.full_text = full_text


    def __str__(self):
        return self.__repr__()
    

    def __repr__(self):
        s = ""
        s += "doc id: {}\n".format(self.doc_id)
        s += "sent id: {}\n".format(self.sent_id)
        s += "text: {}\n".format(" ".join(self.sent))
        s += "event_type: {}\n".format(self.type)
        s += "trigger: {}\n".format(self.trigger['text'])
        for arg in self.args:
            s += "arg {}: {}\n".format(arg['role'], arg['text'])
        s += "----------------------------------------------\n"
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, example_id, feature_id, 
                 enc_text, dec_text,
                 enc_tokens, dec_tokens, 
                 old_tok_to_new_tok_index,  
                 event_type, event_trigger, argument_type,
                 enc_input_ids, enc_mask_ids, 
                 dec_input_ids, dec_mask_ids,
                 answer_text, start_position=None, end_position=None):

        self.example_id = example_id
        self.feature_id = feature_id
        self.enc_text = enc_text
        self.dec_text = dec_text
        self.enc_tokens = enc_tokens
        self.dec_tokens = dec_tokens
        self.old_tok_to_new_tok_index = old_tok_to_new_tok_index
        self.event_type = event_type
        self.event_trigger = event_trigger
        self.argument_type = argument_type
        
        self.enc_input_ids = enc_input_ids
        self.enc_mask_ids = enc_mask_ids
        self.dec_input_ids = dec_input_ids
        self.dec_mask_ids = dec_mask_ids

        self.answer_text = answer_text
        self.start_position = start_position
        self.end_position = end_position


    def __str__(self):
        return self.__repr__()
    

    def __repr__(self):
        s = "" 
        s += "example_id: {}\n".format(self.example_id)
        s += "event_type: {}\n".format(self.event_type)
        s += "trigger_word: {}\n".format(self.event_trigger)
        s += "argument_type: {}\n".format(self.argument_type)
        s += "enc_tokens: {}\n".format(self.enc_tokens)
        s += "dec_tokens: {}\n".format(self.dec_tokens)
        s += "old_tok_to_new_tok_index: {}\n".format(self.old_tok_to_new_tok_index)
        
        s += "enc_input_ids: {}\n".format(self.enc_input_ids)
        s += "enc_mask_ids: {}\n".format(self.enc_mask_ids)
        s += "dec_input_ids: {}\n".format(self.dec_input_ids)
        s += "dec_mask_ids: {}\n".format(self.dec_mask_ids)
        
        s += "answer_text: {}\n".format(self.answer_text)
        s += "start_position: {}\n".format(self.start_position)
        s += "end_position: {}\n".format(self.end_position) 
        return s


class DSET_processor:
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.template_dict, self.argument_dict = self._read_template(self.args.template_path)


    def _read_jsonlines(self, input_file):
        lines = []
        with jsonlines.open(input_file) as reader:
            for obj in reader:
                lines.append(obj)
        return lines


    def _read_json(self, input_file):
        with open(input_file, "r", encoding='utf-8') as f:
            return json.load(f)


    def _read_template(self, template_path):
        template_dict = {}
        argument_dict = {}

        with open(template_path, "r", encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            for line in csv_reader:
                event_type_arg, template = line
                template_dict[event_type_arg] = template
                
                event_type, arg = event_type_arg.split('_')
                if event_type not in argument_dict:
                    argument_dict[event_type] = []
                argument_dict[event_type].append(arg)

        return template_dict, argument_dict

    def _create_example_eeqa(self, lines):
        examples = []
        for line in lines:
            if not line['event']:
                continue
            events = line['event']
            offset = line['s_start']
            full_text = copy.deepcopy(line['sentence'])
            text = line['sentence']
            for event_idx, event in enumerate(events):
                event_type = event[0][1]
                event_trigger = dict()
                start = event[0][0] - offset; end = start+1
                event_trigger['start'] = start; event_trigger['end'] = end
                event_trigger['text'] = " ".join(text[start:end])
                event_trigger['offset'] = offset

                event_args = list()
                for arg_info in event[1:]:
                    arg = dict()
                    start = arg_info[0]-offset; end = arg_info[1]-offset+1
                    role = arg_info[2]
                    arg['start'] = start; arg['end'] = end
                    arg['role'] = role; arg['text'] = " ".join(text[start:end])
                    event_args.append(arg)

                examples.append(ACE_event(event_idx, None, text, event_type, event_trigger, event_args, full_text))
            
        print("{} examples collected.".format(len(examples)))
        # ipdb.set_trace()
        return examples
    
    def _create_example_rams_full_doc(self, lines):
        # maximum doc length is 543 in train (max input ids 803), 394 in dev, 478 in test
        # too long, so we only use sentences contains current trigger and args
        examples = []
        for line in lines:
            doc_key = line["doc_key"]
            if len(line["evt_triggers"]) == 0:
                continue
            
            events = line["evt_triggers"]
            text_tmp = []
            for i, sent in enumerate(line["sentences"]):
                text_tmp += sent

            full_text = copy.deepcopy(list(chain(*line['sentences'])))

            for event_idx, event in enumerate(events):
                event_trigger = dict()
                event_trigger['start'] = event[0]
                event_trigger['end'] = event[1]+1
                event_trigger['text'] = " ".join(text_tmp[event_trigger['start']:event_trigger['end']])

                event_type = event[2][0][0]
                event_trigger['offset'] = 0
                event_args = list()
                for arg_info in line["gold_evt_links"]:
                    if arg_info[0][0] == event[0] and arg_info[0][1] == event[1]:  # match trigger span    
                        evt_arg = dict()
                        evt_arg['start'] = arg_info[1][0]; evt_arg['end'] = arg_info[1][1]+1
                        evt_arg['role'] = arg_info[2].split('arg', maxsplit=1)[-1][2:]
                        evt_arg['text'] = " ".join(text_tmp[evt_arg['start']:evt_arg['end']])
                        event_args.append(evt_arg)

                text=text_tmp
                
                if event_idx > 0:
                    examples.append(ACE_event(doc_key+str(event_idx), None, text, event_type, event_trigger, event_args, full_text))
                else:
                    examples.append(ACE_event(doc_key, None, text, event_type, event_trigger, event_args, full_text))
            
        print("{} examples collected.".format(len(examples)))
        return examples
               
    def _create_example_rams(self, lines):
        # maximum doc length is 543 in train (max input ids 803), 394 in dev, 478 in test
        # too long, so we use a window to cut the sentences.
        W = self.args.window_size
        assert(W%2==0)
        invalid_args_num, all_args_num = 0, 0

        examples = []
        for i, line in enumerate(lines):
            if len(line["evt_triggers"]) == 0:
                continue
            doc_key = line["doc_key"]
            events = line["evt_triggers"]
            assert(len(events)==1)
            event = events[0]

            full_text = copy.deepcopy(list(chain(*line['sentences'])))
            cut_text = list(chain(*line['sentences']))
            sent_length = sum([len(sent) for sent in line['sentences']])
                   
            event_trigger = dict()
            event_trigger['start'] = event[0]
            event_trigger['end'] = event[1]+1
            event_trigger['text'] = " ".join(full_text[event_trigger['start']:event_trigger['end']])
            event_type = event[2][0][0]

            offset, min_s, max_e = 0, 0, W+1
            event_trigger['offset'] = offset
            if sent_length > W+1:
                if event_trigger['end'] <= W//2:     # trigger word is located at the front of the sents
                    cut_text = full_text[:(W+1)]
                else:   # trigger word is located at the latter of the sents
                    offset = sent_length - (W+1)
                    min_s += offset
                    max_e += offset
                    event_trigger['start'] -= offset
                    event_trigger['end'] -= offset 
                    event_trigger['offset'] = offset
                    cut_text = full_text[-(W+1):]

            event_args = list()
            for arg_info in line["gold_evt_links"]:
                if arg_info[0][0] == event[0] and arg_info[0][1] == event[1]:  # match trigger span    
                    all_args_num += 1

                    evt_arg = dict()
                    evt_arg['start'] = arg_info[1][0]
                    evt_arg['end'] = arg_info[1][1]+1
                    evt_arg['text'] = " ".join(full_text[evt_arg['start']:evt_arg['end']])
                    evt_arg['role'] = arg_info[2].split('arg', maxsplit=1)[-1][2:]
                    if evt_arg['start']<min_s or evt_arg['end']>max_e:
                        invalid_args_num += 1
                    else:
                        evt_arg['start'] -= offset
                        evt_arg['end'] -= offset 
                        event_args.append(evt_arg)
            examples.append(ACE_event(doc_key, None, cut_text, event_type, event_trigger, event_args, full_text))
            
        discount_factor = 1-invalid_args_num/all_args_num
        print("{} examples collected.".format(len(examples)))
        print("Discount factor:{}".format(discount_factor))
        return examples

    def _create_example_oneie(self, lines):
        examples = []
        for line in lines:
            doc_id = line['doc_id']
            sent_id = line['sent_id']
            text = line['tokens']
            full_text = copy.deepcopy(line['tokens'])
            entities = {entity["id"]:entity for entity in line['entity_mentions']}
            events = line['event_mentions']
            for event in events:
                event_type = event['event_type']
                event_trigger = event['trigger']
                event_args = event['arguments']
                for arg in event_args:
                    entity_id = arg['entity_id']
                    start, end = entities[entity_id]['start'], entities[entity_id]['end']
                    arg.update({
                        "start": start,
                        "end": end,
                    })

                examples.append(ACE_event(doc_id, sent_id, text, event_type, event_trigger, event_args, full_text))
        
        print("{} examples collected.".format(len(examples)))
        return examples
    
    def _create_example_wikievent_full_doc(self, lines):
        invalid_args_num, all_args_num = 0, 0

        examples = []
        for i, line in enumerate(lines):
            entity_dict = {entity['id']:entity for entity in line['entity_mentions']}
            events = line["event_mentions"]
            if not events:
                continue
            doc_key = line["doc_id"]
            full_text = line['tokens']

            for event in events:
                event_type = event['event_type']
                event_trigger = event['trigger']

                event_trigger['offset'] = 0
                        
                event_args = list()
                for arg_info in event['arguments']:
                    all_args_num += 1

                    evt_arg = dict()
                    arg_entity = entity_dict[arg_info['entity_id']]
                    evt_arg['start'] = arg_entity['start']
                    evt_arg['end'] = arg_entity['end']
                    evt_arg['text'] = arg_info['text']
                    evt_arg['role'] = arg_info['role']
                    event_args.append(evt_arg)
                examples.append(ACE_event(doc_key, None, full_text, event_type, event_trigger, event_args, full_text))

        discount_factor = 1-invalid_args_num/all_args_num
        logger.info("{} examples collected.".format(len(examples)))
        logger.info("Discount factor:{}".format(discount_factor))
        return examples

    def _create_example_wikievent(self, lines):
        W = self.args.window_size
        assert(W%2==0)
        invalid_args_num, all_args_num = 0, 0

        examples = []
        for i, line in enumerate(lines):
            entity_dict = {entity['id']:entity for entity in line['entity_mentions']}
            events = line["event_mentions"]
            if not events:
                continue
            doc_key = line["doc_id"]
            full_text = line['tokens']
            sent_length = len(full_text)

            for event in events:
                event_type = event['event_type']
                cut_text = full_text
                event_trigger = event['trigger']

                offset, min_s, max_e = 0, 0, W+1
                if sent_length > W+1:
                    if event_trigger['end'] <= W//2:     # trigger word is located at the front of the sents
                        cut_text = full_text[:(W+1)]
                    elif event_trigger['start'] >= sent_length-W/2:   # trigger word is located at the latter of the sents
                        offset = sent_length - (W+1)
                        min_s += offset
                        max_e += offset
                        event_trigger['start'] -= offset
                        event_trigger['end'] -= offset 
                        cut_text = full_text[-(W+1):]
                    else:
                        offset = event_trigger['start'] - W//2
                        min_s += offset
                        max_e += offset
                        event_trigger['start'] -= offset
                        event_trigger['end'] -= offset 
                        cut_text = full_text[offset:(offset+W+1)]
                event_trigger['offset'] = offset
                        
                event_args = list()
                for arg_info in event['arguments']:
                    all_args_num += 1

                    evt_arg = dict()
                    arg_entity = entity_dict[arg_info['entity_id']]
                    evt_arg['start'] = arg_entity['start']
                    evt_arg['end'] = arg_entity['end']
                    evt_arg['text'] = arg_info['text']
                    evt_arg['role'] = arg_info['role']
                    if evt_arg['start']<min_s or evt_arg['end']>max_e:
                        invalid_args_num += 1
                        #logger.info(i, min_s, max_e, offset, event_trigger, arg_entity, evt_arg)
                    else:
                        evt_arg['start'] -= offset
                        evt_arg['end'] -= offset 
                        event_args.append(evt_arg)
                examples.append(ACE_event(doc_key, None, cut_text, event_type, event_trigger, event_args, full_text))

        discount_factor = 1-invalid_args_num/all_args_num
        logger.info("{} examples collected.".format(len(examples)))
        logger.info("Discount factor:{}".format(discount_factor))
        return examples

    def create_example(self, file_path):
        if self.args.dataset_type=='ace_eeqa':
            lines = self._read_jsonlines(file_path)
            return self._create_example_eeqa(lines)
        elif self.args.dataset_type=='ace_oneie':
            lines = self._read_jsonlines(file_path)
            return self._create_example_oneie(lines)
        elif self.args.dataset_type=='rams':
            lines = self._read_jsonlines(file_path)
            return self._create_example_rams(lines)
        elif self.args.dataset_type=='rams_full_doc':
            lines = self._read_jsonlines(file_path)
            return self._create_example_rams_full_doc(lines)
        elif self.args.dataset_type=='wikievent':
            lines = self._read_jsonlines(file_path)
            return self._create_example_wikievent(lines)
        else:
            raise NotImplementedError()
    
    def convert_examples_to_features(self, examples):
        features = []
        for (example_idx, example) in enumerate(examples):
            sent = example.sent  
            event_type = example.type
            event_args = example.args
            event_trigger = example.trigger['text']
            trigger_start, trigger_end = example.trigger['start'], example.trigger['end']
            event_args_name = [arg['role'] for arg in event_args]
            enc_text = " ".join(sent)

            old_tok_to_char_index = []     # old tok: split by oneie
            old_tok_to_new_tok_index = []  # new tok: split by BART

            curr = 0
            for tok in sent:
                old_tok_to_char_index.append(curr)
                curr += len(tok)+1
            assert(len(old_tok_to_char_index)==len(sent))

            enc = self.tokenizer(enc_text)
            enc_input_ids, enc_mask_ids = enc["input_ids"], enc["attention_mask"]
            enc_tokens = self.tokenizer.convert_ids_to_tokens(enc_input_ids)  
            while len(enc_input_ids) < self.args.max_enc_seq_length:
                enc_input_ids.append(self.args.pad_token)
                enc_mask_ids.append(self.args.pad_mask_token)
            
            for old_tok_idx, char_idx in enumerate(old_tok_to_char_index):
                new_tok = enc.char_to_token(char_idx)
                old_tok_to_new_tok_index.append(new_tok)    
    
            for arg in self.argument_dict[event_type.replace(':', '.')]:
                dec_text = 'Argument ' + arg + ' in ' + event_trigger + ' event ?' + " "
                     
                dec = self.tokenizer(dec_text)
                dec_input_ids, dec_mask_ids = dec["input_ids"], dec["attention_mask"]
                dec_tokens = self.tokenizer.convert_ids_to_tokens(dec_input_ids) 
                while len(dec_input_ids) < self.args.max_dec_seq_length:
                    dec_input_ids.append(self.args.pad_token)
                    dec_mask_ids.append(self.args.pad_mask_token)
        
                start_position, end_position, answer_text = None, None, None
                if arg in event_args_name:
                    arg_idx = event_args_name.index(arg)
                    event_arg_info = event_args[arg_idx]
                    answer_text = event_arg_info['text']
                    # index before BPE, plus 1 because having inserted start token
                    start_old, end_old = event_arg_info['start'], event_arg_info['end']
                    start_position = old_tok_to_new_tok_index[start_old]
                    end_position = old_tok_to_new_tok_index[end_old] if end_old<len(old_tok_to_new_tok_index) else old_tok_to_new_tok_index[-1]+1 
                else:
                    start_position, end_position = 0, 0
                    answer_text = "__ No answer __"

                feature_idx = len(features)
                features.append(
                      InputFeatures(example_idx, feature_idx, 
                                    enc_text, dec_text,
                                    enc_tokens, dec_tokens,
                                    old_tok_to_new_tok_index,
                                    event_type, event_trigger, arg,
                                    enc_input_ids, enc_mask_ids, 
                                    dec_input_ids, dec_mask_ids,
                                    answer_text, start_position, end_position
                                )
                )
        return features

    
    def convert_features_to_dataset(self, features):

        all_enc_input_ids = torch.tensor([f.enc_input_ids for f in features], \
            dtype=torch.long).to(self.args.device)
        all_enc_mask_ids = torch.tensor([f.enc_mask_ids for f in features], \
            dtype=torch.long).to(self.args.device)
        all_dec_input_ids = torch.tensor([f.dec_input_ids for f in features], \
            dtype=torch.long).to(self.args.device)
        all_dec_mask_ids = torch.tensor([f.dec_mask_ids for f in features], \
            dtype=torch.long).to(self.args.device)
        
        all_start_positions = torch.tensor([f.start_position for f in features], \
            dtype=torch.long).to(self.args.device)
        all_end_positions = torch.tensor([f.end_position for f in features], \
            dtype=torch.long).to(self.args.device)
        all_example_idx = torch.tensor([f.example_id for f in features], \
            dtype=torch.long).to(self.args.device)
        all_feature_idx = torch.tensor([f.feature_id for f in features], \
            dtype=torch.long).to(self.args.device)

        dataset = TensorDataset(all_enc_input_ids, all_enc_mask_ids,
                                all_dec_input_ids, all_dec_mask_ids,
                                all_start_positions, all_end_positions,
                                all_example_idx, all_feature_idx,
                            )
        return dataset


    def load_and_cache_examples(self, file_path, cache_path):
        if not os.path.exists(cache_path) or not self.args.use_cache:
            examples = self.create_example(file_path)
            pickle.dump(examples, open(cache_path, 'wb'))
        else:
            examples = pickle.load(open(cache_path, 'rb'))
        return examples

    
    def load_and_cache_features(self, examples, cache_path):
        if not os.path.exists(cache_path) or not self.args.use_cache:
            features = self.convert_examples_to_features(examples)
            pickle.dump(features, open(cache_path, 'wb'))
        else:
            features = pickle.load(open(cache_path, 'rb'))
        return features
        

    def generate_dataloader(self, set_type):
        assert (set_type in ['train', 'dev', 'test'])
        if not os.path.exists(self.args.cache_path):
            os.makedirs(self.args.cache_path)
        cache_event_path = os.path.join(self.args.cache_path, "{}_events.pkl".format(set_type))
        cache_feature_path = os.path.join(self.args.cache_path, "{}_features.pkl".format(set_type))
        if set_type=='train':
            file_path = self.args.train_file
        elif set_type=='dev':
            file_path = self.args.dev_file
        else:
            file_path = self.args.test_file
        
        examples = self.load_and_cache_examples(file_path, cache_event_path)
        features = self.load_and_cache_features(examples, cache_feature_path)
        dataset = self.convert_features_to_dataset(features)

        if set_type != 'train':
            # Note that DistributedSampler samples randomly
            dataset_sampler = SequentialSampler(dataset)
        else:
            dataset_sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset, sampler=dataset_sampler, batch_size=self.args.batch_size)
        
        return examples, features, dataloader