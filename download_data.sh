# Download the images for each dataset
# wget --show-progress -O attr-ops-data.tar.gz https://www.cs.utexas.edu/~tushar/attribute-ops/attr-ops-data.tar.gz
wget --show-progress -O data/mitstates.zip http://wednesday.csail.mit.edu/joseph_result/state_and_transformation/release_dataset.zip
wget --show-progress -O data/utzap.zip http://vision.cs.utexas.edu/projects/finegrained/utzap50k/ut-zap50k-images.zip
echo "Data downloaded. Extracting files..."

# Dataset metadata, pretrained SVMs and features, 
# tar -zxvf attr-ops-data.tar.gz

# MIT-States
unzip mitstates.zip 'release_dataset/images/*' -d mit-states/
mv mit-states/release_dataset/images mit-states/images/
rm -r mit-states/release_dataset

# UT-Zappos50k
unzip utzap.zip -d ut-zap50k/
mv ut-zap50k/ut-zap50k-images ut-zap50k/images/

echo "Files extracted"