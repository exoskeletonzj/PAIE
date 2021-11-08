for LR in 1e-5 2e-5 3e-5 5e-5
do
    for SEED in 13 21 42 88 100
    do
        work_path=exps/wikievent_single_prompt/$SEED/$LR
        mkdir -p $work_path # make output dir

        COMMAND="python -u engine.py \
        --model_type='base' \
        --dataset_type='wikievent' \
        --model_name_or_path='ckpts/bart-base' \
        --template_path='./data/dset_meta/description_wikievent.csv' \
        --prompt_path './data/prompts/prompts_wikievent_full.csv' \
        --seed=$SEED \
        --output_dir=$work_path \
        --max_steps=20000 \
        --learning_rate=$LR \
        --query_template='arg_trigger' \
        --max_enc_seq_lengt 500 \
        --max_dec_seq_length 20 \
        --max_prompt_seq_length 80 \
        --matching_method_train='max' \
        2>&1 | tee $work_path/log.txt"

        spring.submit arun --gpu -n1 -x SH-IDC1-10-5-30-94 -s "$COMMAND"
    done
done

