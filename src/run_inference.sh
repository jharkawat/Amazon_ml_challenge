CUDA_VISIBLE_DEVICES=0 python run_inference.sh_cls.py \
--trained_model="sample-try-with-t5/finetuned_BERT_epoch_1.model" \
--csv="data/dataset/test.csv" \
--model_type="bert-base-uncased" \
--experiment_name="sample-try-with-bert-base" \
