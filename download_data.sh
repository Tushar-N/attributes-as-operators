cd data/

# Download the images for each dataset
wget --show-progress -O mitstates.zip http://wednesday.csail.mit.edu/joseph_result/state_and_transformation/release_dataset.zip
wget --show-progress -O utzap.zip http://vision.cs.utexas.edu/projects/finegrained/utzap50k/ut-zap50k-images.zip
wget --show-progress -O attribute-ops-data.tar.gz https://www.cs.utexas.edu/~tushar/attribute-ops/attr-ops.tar.gz
echo "Data downloaded. Extracting files..."

# metadata for each dataset
# tar -zxvf attribute-ops-data.tar.gz

# MIT-States
unzip mitstates.zip 'release_dataset/images/*' -d mit-states/
mv mit-states/release_dataset/images mit-states/images/
rm -r mit-states/release_dataset

# UT-Zappos50k
unzip utzap.zip -d ut-zap50k/
mv ut-zap50k/ut-zap50k-images ut-zap50k/images/

echo "Files extracted"