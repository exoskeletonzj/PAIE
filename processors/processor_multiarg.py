import os
import re
import ipdb
import torch
import pickle
import numpy as np

from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from .processor_base import DSET_processor
import random

_PREDEFINED_QUERY_TEMPLATE = {
    "arg_trigger": "Argument: {arg:}. Trigger: {trigger:} ",
    "arg_event": "Argument: {arg:}. Event: {event_type:} ",
    "argonly": "{arg:}",
}


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, example_id, feature_id, 
                event_type, event_trigger,
                enc_text, enc_input_ids, enc_mask_ids, 
                dec_prompt_text, dec_prompt_ids, dec_prompt_mask_ids,
                arg_quries, arg_joint_prompt, target_info,
                old_tok_to_new_tok_index = None, full_text = None, arg_list=None

        ):

        self.example_id = example_id
        self.feature_id = feature_id
        self.event_type = event_type
        self.event_trigger = event_trigger
        
        self.enc_text = enc_text
        self.enc_input_ids = enc_input_ids
        self.enc_mask_ids = enc_mask_ids

        self.dec_prompt_texts = dec_prompt_text
        self.dec_prompt_ids =dec_prompt_ids
        self.dec_prompt_mask_ids=dec_prompt_mask_ids

        if arg_quries is not None:
            self.dec_arg_query_ids = [v[0] for k,v in arg_quries.items()]
            self.dec_arg_query_masks = [v[1] for k,v in arg_quries.items()]
            self.dec_arg_start_positions = [v[2] for k,v in arg_quries.items()]
            self.dec_arg_end_positions = [v[3] for k,v in arg_quries.items()]
            self.start_position_ids = [v['span_s'] for k,v in target_info.items()]
            self.end_position_ids = [v['span_e'] for k,v in target_info.items()]
        else:
            self.dec_arg_query_ids = None
            self.dec_arg_query_masks = None
        
        self.arg_joint_prompt = arg_joint_prompt
        
        self.target_info = target_info
        self.old_tok_to_new_tok_index = old_tok_to_new_tok_index

        self.full_text=full_text
        self.arg_list = arg_list


    def init_pred(self):
        self.pred_dict = dict()
    

    def set_gt(self, model_type):
        self.gt_dict = dict()
        if model_type == 'base':
            for k,v in self.target_info.items():
                # ipdb.set_trace()
                span_s = list(np.where(v["span_s"])[0])
                span_e = list(np.where(v["span_e"])[0])
                self.gt_dict[k] = [(s,e) for (s,e) in zip(span_s, span_e)]
        elif "paie" in model_type:
            for k,v in self.target_info.items():
                self.gt_dict[k] = [(s,e) for (s,e) in zip(v["span_s"], v["span_e"])]
        else:
            assert(0==1)


    def __repr__(self):
        s = "" 
        s += "example_id: {}\n".format(self.example_id)
        s += "event_type: {}\n".format(self.event_type)
        s += "trigger_word: {}\n".format(self.event_trigger)
        s += "old_tok_to_new_tok_index: {}\n".format(self.old_tok_to_new_tok_index)
        
        s += "enc_input_ids: {}\n".format(self.enc_input_ids)
        s += "enc_mask_ids: {}\n".format(self.enc_mask_ids)
        s += "dec_prompt_ids: {}\n".format(self.dec_prompt_ids)
        s += "dec_prompt_mask_ids: {}\n".format(self.dec_prompt_mask_ids)

        return s


class ArgumentExtractionDataset(Dataset):
    def __init__(self, features):
        self.features = features
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]
 
    def collate_fn(self, batch):
        
        enc_input_ids = torch.tensor([f.enc_input_ids for f in batch])
        enc_mask_ids = torch.tensor([f.enc_mask_ids for f in batch])

        if  batch[0].dec_prompt_ids is not None:
            dec_prompt_ids = torch.tensor([f.dec_prompt_ids for f in batch])
            dec_prompt_mask_ids = torch.tensor([f.dec_prompt_mask_ids for f in batch])
        else:
            dec_prompt_ids=None
            dec_prompt_mask_ids=None

        example_idx = [f.example_id for f in batch]
        feature_idx = torch.tensor([f.feature_id for f in batch])

        if batch[0].dec_arg_query_ids is not None:
            dec_arg_query_ids = [torch.LongTensor(f.dec_arg_query_ids) for f in batch]
            dec_arg_query_mask_ids = [torch.LongTensor(f.dec_arg_query_masks) for f in batch]
            dec_arg_start_positions = [torch.LongTensor(f.dec_arg_start_positions) for f in batch]
            dec_arg_end_positions = [torch.LongTensor(f.dec_arg_end_positions) for f in batch]
            start_position_ids = [torch.FloatTensor(f.start_position_ids) for f in batch]
            end_position_ids = [torch.FloatTensor(f.end_position_ids) for f in batch]
        else:
            dec_arg_query_ids = None
            dec_arg_query_mask_ids = None
            dec_arg_start_positions = None
            dec_arg_end_positions = None
            start_position_ids = None
            end_position_ids = None

        target_info = [f.target_info for f in batch]
        old_tok_to_new_tok_index = [f.old_tok_to_new_tok_index for f in batch]

        arg_joint_prompt = [f.arg_joint_prompt for f in batch]
        
        arg_lists = [f.arg_list for f in batch ]

        return enc_input_ids, enc_mask_ids, \
                dec_arg_query_ids, dec_arg_query_mask_ids,\
                dec_prompt_ids, dec_prompt_mask_ids,\
                target_info, old_tok_to_new_tok_index, arg_joint_prompt, arg_lists, \
                example_idx, feature_idx, \
                dec_arg_start_positions, dec_arg_end_positions, \
                start_position_ids, end_position_ids


class MultiargProcessor(DSET_processor):
    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer) 
        self.set_dec_input()
    

    def set_dec_input(self):
        self.arg_query=False
        self.prompt_query=False
        if self.args.model_type == "base":
            self.arg_query = True
        elif "paie" in self.args.model_type:
            self.prompt_query = True
        else:
            raise NotImplementedError(f"Unexpected setting {self.args.model_type}")
     

    def _read_prompt_group(self):
        with open(self.args.prompt_path) as f:
            lines = f.readlines()
        prompts = dict()
        for line in lines:
            if not line:
                continue
            event_type, prompt = line.split(":")
            prompts[event_type] = prompt
        return prompts


    def create_dec_qury(self, arg, event_trigger, event_type):
        dec_text = _PREDEFINED_QUERY_TEMPLATE[self.args.query_template].format(arg=arg, trigger=event_trigger, event_type=event_type)
                
        dec = self.tokenizer(dec_text)
        dec_input_ids, dec_mask_ids = dec["input_ids"], dec["attention_mask"]

        while len(dec_input_ids) < self.args.max_dec_seq_length:
            dec_input_ids.append(self.tokenizer.pad_token_id)
            dec_mask_ids.append(self.args.pad_mask_token)

        matching_result = re.search(arg, dec_text)
        char_idx_s, char_idx_e = matching_result.span(); char_idx_e -= 1
        tok_prompt_s = dec.char_to_token(char_idx_s)
        tok_prompt_e = dec.char_to_token(char_idx_e) + 1

        return dec_input_ids, dec_mask_ids, tok_prompt_s, tok_prompt_e


    def convert_examples_to_features(self, examples):
        if self.prompt_query:
            prompts = self._read_prompt_group()

        if os.environ.get("DEBUG", False): counter = [0, 0, 0]
        features = []
        EXTERNAL_TOKENS = []
        for example_idx, example in enumerate(examples):
            example_id =example.doc_id
            sent = example.sent  
            event_type = example.type
            event_args = example.args
     
            trigger_start, trigger_end = example.trigger['start'], example.trigger['end']
            # NOTE: extend trigger full info in features
            event_trigger = [example.trigger['text'], [trigger_start, trigger_end], example.trigger['offset']]

            event_args_name = [arg['role'] for arg in event_args]
            if os.environ.get("DEBUG", False): counter[2] += len(event_args_name)

            if self.args.context_template == 'with_trigger_sp':
                EXTERNAL_TOKENS = ['<t>', '</t>']
                sent = sent[:trigger_start] + ['<t>'] + sent[trigger_start:trigger_end] + ['</t>'] + sent[trigger_end:]
            
            enc_text = " ".join(sent)

            # change the mapping to idx2tuple (start/end word idx)
            old_tok_to_char_index = []     # old tok: split by oneie
            old_tok_to_new_tok_index = []  # new tok: split by BART

            curr = 0
            for tok in sent:
                if tok not in EXTERNAL_TOKENS:
                    old_tok_to_char_index.append([ curr, curr+len(tok)-1 ]) # exact word start char and end char index
                curr += len(tok)+1

            enc = self.tokenizer(enc_text)
            enc_input_ids, enc_mask_ids = enc["input_ids"], enc["attention_mask"]
            if len(enc_input_ids)> self.args.max_enc_seq_length:
                raise ValueError(f"Please increase max_enc_seq_length above {len(enc_input_ids)}")
            while len(enc_input_ids) < self.args.max_enc_seq_length:
                enc_input_ids.append(self.tokenizer.pad_token_id)
                enc_mask_ids.append(self.args.pad_mask_token)
            
            for old_tok_idx, (char_idx_s, char_idx_e) in enumerate(old_tok_to_char_index):
                new_tok_s = enc.char_to_token(char_idx_s)
                new_tok_e = enc.char_to_token(char_idx_e) + 1
                new_tok = [new_tok_s, new_tok_e]
                old_tok_to_new_tok_index.append(new_tok)    

            # Deal with prompt template
            if self.prompt_query:
                dec_prompt_text = prompts[event_type].strip()
                if dec_prompt_text:
                    dec_prompt = self.tokenizer(dec_prompt_text)
                    dec_prompt_ids, dec_prompt_mask_ids = dec_prompt["input_ids"], dec_prompt["attention_mask"]
                    assert len(dec_prompt_ids)<=self.args.max_prompt_seq_length, f"\n{example}\n{arg_list}\n{dec_prompt_text}"
                    while len(dec_prompt_ids) < self.args.max_prompt_seq_length:
                        dec_prompt_ids.append(self.tokenizer.pad_token_id)
                        dec_prompt_mask_ids.append(self.args.pad_mask_token)
                else:
                    raise ValueError(f"no prompt provided for event: {event_type}")
            else:
                dec_prompt_text, dec_prompt_ids, dec_prompt_mask_ids=None,None,None
                
            arg_list = self.argument_dict[event_type.replace(':', '.')] 
            # NOTE: Large change - original only keep one if multiple span for one arg role
            arg_quries = dict()
            arg_joint_prompt = dict()
            target_info = dict()
            if os.environ.get("DEBUG", False): arg_set=set()
            for arg in arg_list:
                arg_query = None
                prompt_slots = None
                arg_target = {
                    "text": list(),
                    "span_s": list(),
                    "span_e": list()
                }

                if self.arg_query:
                    arg_query = self.create_dec_qury(arg, event_trigger[0], event_type)

                if self.prompt_query:
                    prompt_slots = {
                        "tok_s":list(), "tok_e":list(),
                    }
                    
                    for matching_result in re.finditer(r'\b'+re.escape(arg)+r'\b', dec_prompt_text.split('.')[0]): # Using this more accurate regular expression might further improve rams results
                    # for matching_result in re.finditer(r'\b'+re.escape(arg)+r'\b', dec_prompt_text): # Using this more accurate regular expression might further improve rams results
                        char_idx_s, char_idx_e = matching_result.span(); char_idx_e -= 1
                        tok_prompt_s = dec_prompt.char_to_token(char_idx_s)
                        tok_prompt_e = dec_prompt.char_to_token(char_idx_e) + 1
                        prompt_slots["tok_s"].append(tok_prompt_s);prompt_slots["tok_e"].append(tok_prompt_e)

                answer_texts, start_positions, end_positions = list(), list(), list()
                if arg in event_args_name:
                    # Deal with multi-occurance
                    if os.environ.get("DEBUG", False): arg_set.add(arg)
                    arg_idxs = [i for i, x in enumerate(event_args_name) if x == arg]
                    if os.environ.get("DEBUG", False): counter[0] += 1; counter[1]+=len(arg_idxs)

                    for i, arg_idx in enumerate(arg_idxs):
                        event_arg_info = event_args[arg_idx]
                        answer_text = event_arg_info['text']; answer_texts.append(answer_text)
                        start_old, end_old = event_arg_info['start'], event_arg_info['end']
                        start_position = old_tok_to_new_tok_index[start_old][0]; start_positions.append(start_position)
                        end_position = old_tok_to_new_tok_index[end_old-1][1]; end_positions.append(end_position)

                if self.arg_query:
                    arg_target["span_s"] = [1 if i in start_positions else 0 for i in range(self.args.max_enc_seq_length)]
                    arg_target["span_e"] = [1 if i in end_positions else 0 for i in range(self.args.max_enc_seq_length)]
                    if sum(arg_target["span_s"])==0:
                        arg_target["span_s"][0] = 1
                        arg_target["span_e"][0] = 1
                if self.prompt_query:
                    arg_target["span_s"]= start_positions
                    arg_target["span_e"] = end_positions

                arg_target["text"] = answer_texts
                arg_quries[arg] = arg_query
                arg_joint_prompt[arg] = prompt_slots
                target_info[arg] = arg_target

            if not self.arg_query:
                arg_quries = None
            if not self.prompt_query:
                arg_joint_prompt=None

            # NOTE: one annotation as one decoding input
            feature_idx = len(features)
            features.append(
                    InputFeatures(example_id, feature_idx, 
                                event_type, event_trigger,
                                enc_text, enc_input_ids, enc_mask_ids, 
                                dec_prompt_text, dec_prompt_ids, dec_prompt_mask_ids,
                                arg_quries, arg_joint_prompt, target_info,
                                old_tok_to_new_tok_index = old_tok_to_new_tok_index, full_text=example.full_text, arg_list = arg_list
                    )
            )

        if os.environ.get("DEBUG", False): print('\033[91m'+f"distinct/tot arg_role: {counter[0]}/{counter[1]} ({counter[2]})"+'\033[0m')
        return features

    
    def convert_features_to_dataset(self, features):
        dataset = ArgumentExtractionDataset(features)
        return dataset


    def load_and_cache_examples(self, file_path, cache_path):
        if not os.path.exists(cache_path) or os.environ.get("DEBUG", False) or not self.args.use_cache:
            # print('\033[92m'+'not loading cache examples'+'\033[0m')
            examples = self.create_example(file_path)
            pickle.dump(examples, open(cache_path, 'wb'))
        else:
            examples = pickle.load(open(cache_path, 'rb'))
        return examples

    
    def load_and_cache_features(self, examples, cache_path):
        if not os.path.exists(cache_path) or os.environ.get("DEBUG", False) or not self.args.use_cache:
            # print('\033[92m'+'not loading cache features'+'\033[0m')
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
        dataloader = DataLoader(dataset, sampler=dataset_sampler, batch_size=self.args.batch_size, collate_fn=dataset.collate_fn)

        return examples, features, dataloader