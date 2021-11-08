
for LR in 1e-5 2e-5 3e-5 5e-5
do
    for SEED in 13 21 42 88 100
    do
        work_path=exps/ace05_fineprompt/$SEED/${LR}
        mkdir -p $work_path # make output dir

        COMMAND="python -u engine.py \
        --model_type='paie' \
        --dataset_type='ace_eeqa' \
        --model_name_or_path='ckpts/bart-base' \
        --template_path='./data/dset_meta/description_ace.csv' \
        --prompt_path './data/prompts/prompts_ace_full_and_def.csv' \
        --seed=$SEED \
        --output_dir=$work_path  \
        --learning_rate=$LR \
        --batch_size 16 \
        --eval_steps 200  \
        --max_steps=10000 \
        --max_enc_seq_lengt 180 \
        --max_dec_seq_length 20 \
        --max_prompt_seq_length 200 \
        2>&1 | tee $work_path/log.txt"

        spring.submit arun --gpu -n1 -s "$COMMAND"
    done
done


