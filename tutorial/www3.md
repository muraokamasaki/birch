# Tutorial

This tutorial will go through the steps that we take to perform document ranking on the NTCIR WWW3 task.
We will be using BERT-base fine-tuned on TREC Microblog (MB).

 
## Environment & Data

We first set up the enviroment. Conda is also an alternative to virtualenv. Tested using python 3.5.
```
# Set up environment
pip install virtualenv
virtualenv -p python3.5 birch_env
source birch_env/bin/activate

# Install dependencies
pip install Cython  # jnius dependency
pip install -r requirements.txt
```

One thing to note here is that we use `torch=1.2.0`, thus having the right version of cuda is important. Check here <https://pytorch.org/get-started/previous-versions/#v120>.
```
# For inference, the Python-only apex build can also be used
git clone https://github.com/NVIDIA/apex
cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# Set up Anserini (last reproduced with commit id: 5da46f610435be6364700bc5a6144253ed3f3b59)
git clone https://github.com/castorini/anserini.git
cd anserini && mvn clean package appassembler:assemble

# Download data and models
wget https://zenodo.org/record/3381673/files/emnlp_bert4ir_v2.tar.gz
tar -xzvf emnlp_bert4ir_v2.tar.gz

cd eval && tar xvfz trec_eval.9.0.4.tar.gz && cd trec_eval.9.0.4 && make && cd ../..
mkdir logs
```

As of writing, Anserini does not have the `/eval` directory by default, and so we make a copy.
```
cp -r eval anserini/eval
```

## Training

We plan to fine tune BERT-base on MB, but it is possible to use BERT-large as well. (Using other collections is also possible, but only the MB dataset is provided.)
Batch size should be changed as necessary (originally 16).
```
export CUDA_VISIBLE_DEVICES=0; experiment=base_mb; \
nohup python -u src/main.py --mode training --experiment ${experiment} --collection mb \
--batch_size 4 --data_path data --model_path models/saved.${experiment} --eval_steps 1000 \
--device cuda --output_path logs/out.${experiment} > logs/${experiment}.log 2>&1 &
```


## Inference

We conduct inference on robust04 (although core17 and core18 can alternatively be used).
This will generate a file `data/predictions/predict.${experiment}` that will be used in the next step.
```
nohup python -u src/main.py --mode inference --experiment ${experiment} --collection robust04 \
--load_trained --model_path models/saved.${experiment}_1 --batch_size 4 --data_path data \
--predict_path data/predictions/predict.${experiment} --device cuda \
--output_path logs/out.${experiment} > logs/${experiment}.log 2>&1 &
```

If we want to use the models provided by the authors (check the `models` directory), we can start from here.
Make sure to set the `${experiment}` parameter to match the prediction suffix found in `data/predictions` (for example msmarco_mb_robust04).
The next steps tune the hyperparameters and evaluate the BERT model on robust04.
```
./eval_scripts/train.sh ${experiment} robust04 anserini

./eval_scripts/test.sh ${experiment} robust04 anserini

./eval_scripts/eval.sh ${experiment} robust04 anserini data

# Use the following if you want ndcg@20 instead of ndcg@10
./eval_scripts/eval.sh ${experiment} robust04 anserini data 20
```

We should get the following scores (BERT-base, MB):
||||
|-|-|-|
|1S:|map|0.3204|
||P@20|0.4163|
||ndcg@10|0.4824|
||ndcg@20|0.4691|
|2S:|map|0.3204|
||P@20|0.4179|
||ndcg@10|0.4842|
||ndcg@20|0.4715|
|3S:|map|0.3203|
||P@20|0.4179|
||ndcg@10|0.4862|
||ndcg@20|0.4722|

## Retrieve sentences from top candidate documents

From here on, we will work on the NTCIR WWW3 test collection.
You need to have indexed the ClueWeb12B corpus to do this step. We retrieve the top 1000 documents for each topic using bm25.
You also need the www3 topics saved as `data/topics/topics.www3.txt`.
This will create a file called `data/datasets/www3_sents.csv` that contains the sentences from the top candidate documents.
```
python src/utils/split_docs.py --collection <robust04, core17, core18> \
--index <path/to/index> --data_path data --anserini_path <path/to/anserini/root>
```

## Predictions

We use BERT to score the sentences generated in the previous step.
This step will generate the file `data/predictions/predict.${experiment}`, which will overwrite the predictions file generated in an earlier step (from the inference of robust04). If you want to keep the (robust04) file, you should make a copy of it, although it is not needed in future steps.
The same can be said for the `runs/` directory which contains the overall ranked predictions for robust04. It should be renamed if you intend to keep it, although it is not needed in future steps.
```
nohup python -u src/main.py --mode inference --experiment ${experiment} --collection www3 \
--load_trained --model_path models/saved.${experiment}_1 --batch_size 4 --data_path data \
--predict_path data/predictions/predict.${experiment} --device cuda \
--output_path logs/out.${experiment} > logs/${experiment}.log 2>&1 &
```

Generate folds for the WWW3 test collection. Since we are doing prediction, we put all topics in one fold.
```
python src/utils/folds www3
```


Generate the final predictions for WWW3. The files are in the `runs/` folder, named `runs.mb.cv.a`, `runs.mb.cv.ab`, and `runs.mb.cv.abc`, which are the runs using the top 1, 2, or 3 sentences respectively (read the research paper for more information).
```
./eval_scripts/predict.sh ${experiment} www3 anserini
```

At this point, we the runs are ready for submission (although they might need to be reformatted to fit the NTCIR guidelines). Since the WWW3 task it over, we can evaluate it using the official qrels. It should be saved as `data/qrels/qrels.www3.txt`, and formatted according to the TREC guidelines.
```
./eval_scripts/eval.sh ${experiment} www3 anserini data
```

We should get the following scores (BERT-base, MB):
||||
|-|-|-|
|1S:|map|0.3808|
||P@20|0.7719|
||ndcg@10|0.5964|
|2S:|map|0.3855|
||P@20|0.7856|
||ndcg@10|0.6135|
|3S:|map|0.3855|
||P@20|0.7831|
||ndcg@10|0.6125|

The results are not particularly impressive. However if we repeat the experiment using the provided model (BERT-large and fine tuned on MSMARCO-\>MB), we achieve much better results.

||||
|-|-|-|
|1S:|map|0.3676|
||P@20|0.8094|
||ndcg@10|0.6828|
|2S:|map|0.3627|
||P@20|0.8169|
||ndcg@10|0.6885|
|3S:|map|0.3648|
||P@20|0.8094|
||ndcg@10|0.6930|
