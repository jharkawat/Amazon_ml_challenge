# Amazon_ml_challenge
Code base of for Product Browse Node Classification 
                        organised by [Amazon India](https://www.hackerearth.com/challenges/competitive/amazon-ml-challenge/problems/)  

**Team Name**: *off-by-one-bit* (Top 5% among 3000+ teams) 

## Dependencies and setup

| Dependency | Version | Installation Command |
| ---------- | ------- | -------------------- |
| Python     | 3.8     | `conda create --name covid_entities python=3.8` and `conda activate covid_entities` |
| PyTorch, cudatoolkit    | 1.5.0, 10.1   | `conda install pytorch==1.5.0 cudatoolkit=10.1 -c pytorch` |
| Transformers 🤗 (Huggingface) | 2.9.0 | `pip install transformers==2.9.0` |
| Scikit-learn | 0.23.1 | `pip install scikit-learn==0.23.1` |
| scipy        | 1.5.0  | `pip install scipy==1.5.0` |
| NLTK    | 3.5  | `pip install nltk==3.5` |

<!--
- python 3.8
```conda create --name covid_entities python=3.8``` & ```conda activate covid_entities```
- PyTorch 1.5.0, cudatoolkit=10.1
```conda install pytorch==1.5.0 cudatoolkit=10.1 -c pytorch```
- Huggingface transformers - 2.9.0
```pip install transformers==2.9.0```
- scikit-learn 0.23.1
```pip install scikit-learn==0.23.1```
- scipy 1.5.0
```pip install scipy==1.5.0```
- ekphrasis 0.5.1
```pip install nltk==3.5```

-->

## Pretrained Models

Pretrained Weights       | Links
------------- | -------------
**Encoded Pickles of datasets**  | [here](https://github.com/tejasvaidhyadev/off-by-one-bit/releases/download/V0.1/encoded_data_test.pkl)  
**Pretrained distilbert model**  | [here](https://github.com/tejasvaidhyadev/off-by-one-bit/releases/download/v0.2/sample-distilbert-run2.zip)
**Pretrained RoBERTa with Preprocessing** (Training dataset: 70%) | [here](https://github.com/tejasvaidhyadev/off-by-one-bit/releases/download/v0.3/sample-roberta-training.zip)
**Pretrained RoBERTa with Preprocessing**  (Full Training data) | [here](https://github.com/tejasvaidhyadev/off-by-one-bit/releases/download/v0.4/all-data-12h-8gpu-run4.zip)
 **Server logs**  | [here](https://github.com/tejasvaidhyadev/off-by-one-bit/releases/download/v0.42/server_log.zip)
 
### Note!   
> Encoded Pickles are tokenised dataset embeddding for fast loading


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

Pretrained Weights       | Links
------------- | -------------
**Distilbert**  | 59.21 %  
**Roberta**  | 67.23 % 


## Team Members
Tejas Vaidhya | Jalaj Harkawat | Hardik Aggarwal | Apoorve Singhle
(Order randomly generated)

# Miscellanous
- **License**: MIT
- You may contact us by opening an issue on this repo. Please allow 2-3 days of time to address the issue.
- For query related to dataset, Please contact [@Hackerearth](https://www.hackerearth.com/challenges/competitive/amazon-ml-challenge/problems/) or [@Amazon ML India](https://amazonscienceindia.splashthat.com/)
