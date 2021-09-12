work_path=exps/rams_full_doc
mkdir -p $work_path # make output dir

PYTHONPATH=./ python -u engine.py --model_type='paie' \
--dataset_type='rams_full_doc' --eval_steps 500 --seed=$RANDOM \
--output_dir=$work_path --max_steps=20000 --learning_rate=0.000015 \
--template_path='./data/dset_meta/description_rams.csv' --model_name_or_path='ckpts/bart-large' \
--query_template='arg_prompt' --context_template='with_trigger_sp' --prompt_context='decoder' \
--max_enc_seq_lengt 805 --max_dec_seq_length 20 --max_prompt_seq_length 50 \
--batch_size 2 --matching_method_train='max' \
--prompt_type group --prompt_path './data/prompts/prompts_rams_full.csv' \
2>&1 | tee $work_path/log.txt

