# syntax:
# python train.py --dataset <mitstates|zappos>
# 				--data_dir </path/to/dataset/>
# 				--glove_init or --clf_init
# 				--model <visprodNN|redwine|labelembed+|attributeop>
# 				--cv_dir </path/to/save/checkpoints/>
# 				--static_inp (for redwine)
# 				--lambda_<aux|inv|comm|ant> <weight>


# Train the VisProdNN baseline model on MIT-States
python train.py --dataset mitstates --data_dir data/mit-states/ --batch_size 512 --lr 1e-4 --max_epochs 800 --model visprodNN --cv_dir cv/mitstates/visprodNN

# Train the RedWine model on MIT-States 
python train.py --dataset mitstates --data_dir data/mit-states/ --batch_size 512 --lr 1e-4 --max_epochs 800 --clf_init --model redwine --cv_dir cv/mitstates/redwine --static_inp

# Train the LabelEmbed model on UT-Zappos
python train.py --dataset zappos --data_dir data/ut-zap50k/ --batch_size 512 --lr 1e-4 --max_epochs 1000 --glove_init --model redwine --cv_dir cv/zappos/labelembed --static_inp

# Train the LabelEmbed+ model on UT-Zappos (with 2 layers)
python train.py --dataset zappos --data_dir data/ut-zap50k/ --batch_size 512 --lr 1e-4 --max_epochs 1000 --glove_init --model labelembed+ --nlayers 2 --cv_dir cv/zappos/labelembed+

# Train an Attribute Operator model on MIT-States with only the auxiliary regularizer
python train.py --dataset mitstates --data_dir data/mit-states/ --batch_size 512 --lr 1e-4 --max_epochs 800 --glove_init --model attributeop --cv_dir cv/mitstates/attrop+aux --lambda_aux 1000.0

# Train an Attribute Operator model on MIT-States with the best regularizers
python train.py --dataset mitstates --data_dir data/mit-states/ --batch_size 512 --lr 1e-4 --max_epochs 800 --glove_init --model attributeop --cv_dir cv/mitstates/attrop+aux+inv+comm --lambda_aux 1000.0 --lambda_inv 1.0 --lambda_comm 1.0

# Train an Attribute Operator model on Zappos with the best regulairzers
python train.py --dataset zappos --data_dir data/ut-zap50k/ --batch_size 512 --lr 1e-4 --max_epochs 1000 --glove_init --model attributeop --cv_dir cv/zappos/attrop+aux+comm --lambda_aux 1.0 --lambda_comm 1.0


# Train AnalogousAttr models using BPTF
cd tensor-completion
matlab -nodisplay -nodesktop -r "try; complete incomplete/mitstates completed/mitstates_30_50.mat 30 50 1; catch; end; quit" > log.log 2> log.err
matlab -nodisplay -nodesktop -r "try; complete incomplete/zappos completed/zappos_100_50.mat 100 50 2; catch; end; quit" > log.log 2> log.err