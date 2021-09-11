import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--model_name_or_path", default="./ckpts/bart-base", type=str)

    parser.add_argument("--dataset_type", type=str)
    parser.add_argument("--cache_path", default='./data/cached_multiarg_new', type=str)
    parser.add_argument("--template_path", default='./data/description_queries_rams.csv', type=str)
    parser.add_argument("--encoder_template_path", default='./data/RAMS_relation_template.csv', type=str)
    parser.add_argument("--prompt_path", default='./data/prompts_rams_full.csv', type=str)

    parser.add_argument("--output_dir", default='./outputs_res', type=str)
    
    parser.add_argument("--pad_token", default=1, type=int)
    parser.add_argument("--pad_mask_token", default=0, type=int)
    parser.add_argument("--device", default='cuda', type=str)
    parser.add_argument("--max_enc_seq_length", default=500, type=int)
    parser.add_argument("--max_dec_seq_length", default=20, type=int)
    parser.add_argument("--max_prompt_seq_length", default=64, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=5.0, type=float)
    parser.add_argument("--max_span_length", default=10, type=int)
    parser.add_argument("--max_span_num", default=1, type=int)
    parser.add_argument("--max_steps", default=10000, type=int)
    parser.add_argument("--warmup_steps", default=0.1, type=float)
    parser.add_argument('--logging_steps', default=100, type=int)
    parser.add_argument('--eval_steps', default=200, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--use_cache', default=False, action="store_true")

    # Window size for document level dset (rams)
    parser.add_argument("--window_size", default=250, type=int)

    # paie setting
    parser.add_argument('--matching_method_train', default="naive", choices=["naive", "max", 'accurate'], type=str)

    # input text setting
    parser.add_argument('--context_template', default="with_trigger_sp", 
                        choices=["ori", "with_trigger_sp"],type=str)
    parser.add_argument('--query_template', default="arg_trigger", 
                        choices=["arg_trigger", "arg_event", "argonly", "arg_prompt"],type=str)
    parser.add_argument('--encoder_template', default="no_encoder_template",
                        choices=["no_encoder_template", "encoder_template"],type=str)

    # use start token or mean pooling to get query vector
    parser.add_argument('--prompt_type', default="group",
                        choices=["twin", "group"], type=str)
    parser.add_argument('--prompt_context', default="decoder",
                        choices=['encoder', 'decoder'], type=str)
      
    args = parser.parse_args()
    return args