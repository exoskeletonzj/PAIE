for LR in 1e-5 2e-5 3e-5 5e-5
do
    for SEED in 13 21 42 88 100
    do
        work_path=exps/wikievent_softprompt/$SEED/$LR
        mkdir -p $work_path

        COMMAND="python -u engine.py \
        --model_type='paie' \
        --dataset_type='wikievent' \
        --model_name_or_path='ckpts/bart-base' \
        --template_path='./data/dset_meta/description_wikievent.csv' \
        --prompt_path './data/prompts/prompts_wikievent_continuous.csv' \
        --seed=$SEED \
        --output_dir=$work_path \
        --learning_rate=$LR \
        --max_steps=20000 \
        --max_enc_seq_lengt 500 \
        --max_dec_seq_length 20 \
        --max_prompt_seq_length 80 \
        2>&1 | tee $work_path/log.txt"

        spring.submit arun --gpu -n1 -x SH-IDC1-10-5-30-94 -s "$COMMAND"
    done
done