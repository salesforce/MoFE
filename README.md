# MoFE: Mixture of Factual Experts for Controlling Hallucinations in Abstractive Summarization

**Paper Link:** https://arxiv.org/abs/2110.07166

**Authors:** Prafulla Kumar Choubey, Jesse Vig, Wenhao Liu, Nazneen Fatema Rajani

## Install dependencies
Use the provided Dockerfile to build docker image.

## Running Code
Download pre-trained [DAE evaluation models](https://github.com/tagoyal/factuality-datasets). We used ```DAE_xsum_human_best_ckpt``` model for XSUM and ```ENT-C_dae``` for CNN_DM datasets.
1. **Data Preparation:** Specify ```--metric``` as one of the ```dae/ner-p/ner-r```. Assuming ```train.source``` and ```train.target``` contain source articles and summaries, run:
    1. *Unfiltered* (used for model-based expert training)
        ```
        python prepare_data.py \
            --metric dae
            --source_doc $DATA_DIR/train.source \
            --target_summary $DATA_DIR/train.target \
            --dump_csv $DATA_DIR/train.csv \
        ```
    2. *Filtered* (used for reference-based expert training)
        ```
        python data_filtering.py \
            --metric dae
            --source_doc $DATA_DIR/train.source \
            --target_summary $DATA_DIR/train.target \
            --dump_csv $DATA_DIR/train-unhallucinated.csv \
            --dae_model $PRETRAINED_DAE_EVALUATION_MODEL
        ```
2. **Training:** 
    1. We used ```facebook/bart-large-xsum```, ```facebook/bart-large-cnn``` and ```google/pegasus-xsum``` models in our experiments.
    2. *kl_alpha:* Loss is defined as ```kl_alpha * KL divergence loss + (1-kl_alpha) * RL loss```. 
    3. *regularization_model:* set to ```True``` for model-based expert training and ```False``` for reference-based expert training.
    4. *default_max_length:* set to ```False``` if you want to restrict sampled summary length to the length of the longest reference summary in a given batch. When a model is trained with ```ner-r``` reward, it may start generating very longer summaries to improve recall. In our experiments, we set ```default_max_length``` to ```False``` when training NER-R expert on the CNN_DM dataset. In all other experiments, we set ```default_max_length``` to ```True```.
    5. *reward_metric:* Choose one of the ```dae/ner-p/ner-r```.
        ```
        python train_expert.py \
            --output_dir $OUTPUT_MODEL/xsum-pegasus-dae-reference
            --do_train \
            --do_eval \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 8 \
            --predict_with_generate \
            --remove_unused_columns False \
            --gradient_accumulation_steps 4 \
            --num_train_epochs 1 \
            --training_file $DATA_DIR/train-unhallucinated.csv 
            --kl_alpha 0.9 \
            --model_checkpoint_name google/pegasus-xsum \
            --cache_dir_path $PRETRAINED_MODELS/pegasus-xsum \
            --regularization_model False \
            --reward_metric dae \
            --dae_model_dir $PRETRAINED_DAE_EVALUATION_MODEL \
            --default_max_length True
        ```
3. **Weights Ensembling:**
    1. *alphas:* ```,``` separated values. Number of alpha values should be same as the number of epxerts. Alpha for the base model is calculated as ```1.0 - sum(alphas)```.
    2. *experts:* ```,``` separated paths for expert models.
    ```
    python weights_ensemble.py \
        --baseline_checkpoint_name facebook/bart-large-xsum
        --baseline_cache_dir_path $PRETRAINED_MODELS/bart-large-xsum
        --alphas 0.5,0.1,0.1
        --experts $OUTPUT_MODEL/xsum-bart-dae-model/,$OUTPUT_MODEL/xsum-bart-ner-p-model/,$OUTPUT_MODEL/xsum-bart-ner-r-model/
        --mofe_weight_ensemble_path $OUTPUT_MODEL/MoFE_weight_ensemble
    ```
    To generate summary:
    ```
    python generate_summary.py \
      --eval_data_path $DATA_DIR/xsum-val.csv \
      --dump_file $OUTPUT_SUMMARY/weights_bart_xsum.txt \
      --model_path $OUTPUT_MODEL/MoFE_weight_ensemble \
      --batch_size 16
    ```
   
3. **Logits Ensembling:**
   ```
   python new_logits_ensemble.py \
      --experts $OUTPUT_MODEL/xsum-bart-dae-model/,$OUTPUT_MODEL/xsum-bart-ner-p-model/,$OUTPUT_MODEL/xsum-bart-ner-r-model \
      --alphas 0.5,0.1,0.1 \
      --eval_data_path $DATA_DIR/xsum-val.csv \
      --num_beams 6 \
      --output_path $OUTPUT_SUMMARY/logits_bart_xsum.txt \
      --per_device_eval_batch_size 3
   ```



## References
This repo uses DAE factuality metric-based reward to train experts. We have included the relevant code from the [repo](https://github.com/tagoyal/factuality-datasets).