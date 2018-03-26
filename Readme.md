# Attributes as Operators

![Attributes as Operators Model](https://user-images.githubusercontent.com/4995097/37882661-ed5749b0-306a-11e8-83f2-6e566316f660.png)

This code implements an embedding learning framework for visual attribute-object composition (e.g. sliced+orange = sliced orange) by treating objects as vectors, and attributes as operators that modify the object vectors to appropriately compose them into a complex concept. Not only does our approach align conceptually with the linguistic role of attributes as modifiers, but it also generalizes to recognize unseen compositions of objects and attributes. Our model recognizes unseen compositions robustly in an open-world setting on two challenging datasets, and can also generalize to compositions where objects themselves were unseen during training.

This is the code accompanying the work:  
Tushar Nagarajan and Kristen Grauman. Attributes as Operators [[arxiv]](https://arxiv.org/pdf/???)

## Prerequisites
The code is written and tested using Python (2.7) and PyTorch (v0.3.0). MATLAB is required for the AnalogousAttr models only.

**Packages**: Install using `pip install -r requirements.txt`

**Datasets and Features**: We include a script to download all the necessary data: images, features and metadata for the two datasets, pretrained SVM classifier weights and tensor completion code. It must be run before training the models.
```bash
bash download_data.sh
```

## Training a model

Models can be trained using the train script with parameters for the model (visprodNN, redwine, labelembed+, attributeop) and the various regularizers (aux, inv, comm, ant). For example, to train the LabelEmbed+ baseline on UT-Zappos:
```bash
python train.py --dataset zappos --data_dir data/ut-zap50k/ --batch_size 512 --lr 1e-4 --max_epochs 1000 --glove_init --model labelembed+ --nlayers 2 --cv_dir cv/zappos/labelembed+
```


To train the AttributeOperator model with all regularizers on MIT-States and UT-Zappos:
```bash
python train.py --dataset mitstates --data_dir data/mit-states/ --batch_size 512 --lr 1e-4 --max_epochs 800 --glove_init --model attributeop --cv_dir cv/mitstates/attrop+aux+inv+comm --lambda_aux 1000.0 --lambda_inv 1.0 --lambda_comm 1.0
python train.py --dataset zappos --data_dir data/ut-zap50k/ --batch_size 512 --lr 1e-4 --max_epochs 1000 --glove_init --model attributeop --cv_dir cv/zappos/attributeop --lambda_aux 1.0 --lambda_comm 1.0
```

The AnalogousAttr model can be trained using the MATLAB scripts in `tensor-completion/`. This will use BPTF to complete the incomplete tensors generated from pretrained SVM weights.
```bash
cd tensor-completion
matlab -nodisplay -nodesktop -r "try; complete incomplete/mitstates completed/mitstates_30_50.mat 30 50 1; catch; end; quit" > log.log 2> log.err
matlab -nodisplay -nodesktop -r "try; complete incomplete/zappos completed/zappos_100_50.mat 100 50 2; catch; end; quit" > log.log 2> log.err
```
 
More examples to train various kinds of models can be found in `examples.sh`. 



## Model Evaluation

The Embedding models can be evaluated using the test script.
```bash
python test.py --model attributeop --lambda_aux 1.0 --load cv/mitstates/attrop+aux+inv+comm/ckpt_E_800_At_0.188_O_0.227_Cl_0.120_Op_0.114.t7
```

The AnalogousAttr model can be evaluated using:
```bash
python svm.py --dataset mitstates --data_dir data/mit-states/ --evaltf --completed tensor-completion/completed/mitstates_30_50.mat
python svm.py --dataset zappos --data_dir data/ut-zap50k/ --evaltf --completed tensor-completion/completed/zappos_100_50.mat
```

## Cite

If you find this repository useful in your own research, please consider citing:
```
???
```
