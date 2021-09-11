work_path=exps/wikievent
mkdir -p $work_path # make output dir

CUDA_VISIBLE_DEVICES=6 PYTHONPATH=./ python -u engine.py --model_type='paie' \
--dataset_type='wikievent' --eval_steps 500 --seed=$RANDOM \
--output_dir=$work_path --max_steps=20000 --learning_rate=0.000015 \
--template_path='./data/dset_meta/description_wikievent.csv' --model_name_or_path='ckpts/bart-large' \
--query_template='arg_prompt' --context_template='with_trigger_sp' --prompt_context='decoder' \
--max_enc_seq_lengt 500 --max_dec_seq_length 20 --max_prompt_seq_length 80 \
--batch_size 4 --matching_method_train='max' \
--prompt_type group --prompt_path './data/prompts/prompts_wikievent_full.csv' \
2>&1 | tee $work_path/log.txt &

