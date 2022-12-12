# Download everything
wget --show-progress -O data/attr-ops-data.tar.gz https://utexas.box.com/shared/static/h7ckk2gx5ykw53h8o641no8rdyi7185g.gz
wget --show-progress -O data/mitstates.zip http://wednesday.csail.mit.edu/joseph_result/state_and_transformation/release_dataset.zip
wget --show-progress -O data/utzap.zip http://vision.cs.utexas.edu/projects/finegrained/utzap50k/ut-zap50k-images.zip
wget --show-progress -O tensor-completion/bptf.tar.bz https://www.cs.cmu.edu/~lxiong/bptf/export_bptf.tar.bz
echo "Data downloaded. Extracting files..."

# Dataset metadata, pretrained SVMs and features, tensor completion data
tar -zxvf data/attr-ops-data.tar.gz --strip 1

# tensor completion code
tar -xvf tensor-completion/bptf.tar.bz -C tensor-completion/ --strip 1

# dataset images
cd data/

# MIT-States
unzip mitstates.zip 'release_dataset/images/*' -d mit-states/
mv mit-states/release_dataset/images mit-states/images/
rm -r mit-states/release_dataset
rename "s/ /_/g" mit-states/images/*

# UT-Zappos50k
unzip utzap.zip -d ut-zap50k/
mv ut-zap50k/ut-zap50k-images ut-zap50k/_images/

cd ..
python utils/reorganize_utzap.py

# remove all zip files and temporary files
rm -r data/attr-ops-data.tar.gz data/mitstates.zip data/utzap.zip data/ut-zap50k/_images
