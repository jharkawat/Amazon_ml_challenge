# off-by-one-bit
Code base for Product Browse Node Classification 
                        - by [Amazon India](https://www.hackerearth.com/challenges/competitive/amazon-ml-challenge/problems/)
- Supports Multi-GPU
- Huggingface Model Card
# Pretrained Weights
- **Encoded Pickles**: [Link](https://github.com/tejasvaidhyadev/off-by-one-bit/releases/download/V0.1/encoded_data_test.pkl)  
    Fast load tokenized embeddding using pickle file
- **Pretrained distilbert model**: [Link](https://github.com/tejasvaidhyadev/off-by-one-bit/releases/download/v0.2/sample-distilbert-run2.zip)

- **Pretrained RoBERTa with Preprocessing** (training data: 70%): [Link](https://github.com/tejasvaidhyadev/off-by-one-bit/releases/download/v0.3/sample-roberta-training.zip)
- **Pretrained RoBERTa with Preprocessing** (full data): [Link](https://github.com/tejasvaidhyadev/off-by-one-bit/releases/download/v0.4/all-data-12h-8gpu-run4.zip)
- **Server logs**: [link](https://github.com/tejasvaidhyadev/off-by-one-bit/releases/download/v0.42/server_log.zip)

# Instruction for running code

**Setup Instruction**
```
cd src/
conda env create -f env.yml
bash download_file.sh
```
**1.** **preprocessing**
```
python splitter.py
```

**2.** **Training Mode**
```
python src/train.py --model {hugging_face_model card}  --experiment_name {experiment_name} 
--used_tokenized_data {tokenized_pickle_file} --epochs {num_epochs} --batch_size_train {batch_size} 
--batch_size_val {batch_size_val} --dummy {dummy_data_checked} --full_finetuning {layer_of_finetuning} 
--loading_from_prev_pretrain {path_to_previous_pretrain} --trained_model {trained_model_path}
```

**3.** **Testing Mode**
```
python src/test.py --csv {path_to_file}
```
**4.** **Inference Mode**
```
python src/inference.py
--trained_model {saved_model_name_from_huggingface}
--csv {csv_of_test_sets}
--model_type {"output_directory"}
--experiment_name {sample-distilbert-run2}
```
**Accuracy on test set:** 

Distilbert: 59.21 %  
Roberta: 67.23 %  

## Team Members
Tejas Vaidhya | Jalaj Harkawat | Hardik Aggarwal | Apoorve Singhle
(Order randomly generated)

# Miscellanous
- **License**: MIT
- ToDo Documentation (Coming soon)

