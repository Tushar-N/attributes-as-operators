# Download everything
wget --show-progress -O data/attr-ops-data.tar.gz https://www.cs.utexas.edu/~tushar/attribute-ops/attr-ops-data.tar.gz
wget --show-progress -O data/mitstates.zip http://wednesday.csail.mit.edu/joseph_result/state_and_transformation/release_dataset.zip
wget --show-progress -O data/utzap.zip http://vision.cs.utexas.edu/projects/finegrained/utzap50k/ut-zap50k-images.zip
wget --show-progress -O tensor-completion/bptf.tar.bz https://www.cs.cmu.edu/~lxiong/bptf/export_bptf.tar.bz
echo "Data downloaded. Extracting files..."

# Dataset metadata, pretrained SVMs and features, tensor completion data
tar -zxvf data/attr-ops-data.tar.gz

# tensor completion code
cd tensor-completion
tar -xvf bptf.tar.bz
mv export_bptf/* .
rm -r export_bptf
cd ..

# dataset images
cd data/

# MIT-States
unzip mitstates.zip 'release_dataset/images/*' -d mit-states/
mv mit-states/release_dataset/images mit-states/images/
rm -r mit-states/release_dataset
rename "s/ /_/g" mit-states/images/*

# UT-Zappos50k
unzip utzap.zip -d ut-zap50k/
mv ut-zap50k/ut-zap50k-images ut-zap50k/images/
