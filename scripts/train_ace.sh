work_path=exps/ace05
mkdir -p $work_path # make output dir

CUDA_VISIBLE_DEVICES=4 PYTHONPATH=./ python -u engine.py --model_type='paie' \
--dataset_type='ace_eeqa' --eval_steps 500  --max_steps=10000 \
--seed=$RANDOM --output_dir=$work_path  --learning_rate=0.00002 \
--template_path='./data/dset_meta/description_ace.csv' \
--model_name_or_path='ckpts/bart-large' --query_template='arg_prompt' \
--context_template='with_trigger_sp' --prompt_context='decoder' \
--max_enc_seq_lengt 180 --max_dec_seq_length 20 --max_prompt_seq_length 50 \
--batch_size 24 --matching_method_train='max' \
--prompt_type group --prompt_path './data/prompts/prompts_ace_full.csv' \
2>&1 | tee $work_path/log.txt &


