import cPickle as pickle
import collections
import scipy.io
import numpy as np

data_dir = '/scratch/cluster/tushar/datasets/ut-zap50k/'
meta_dir = data_dir + 'ut-zap50k-data/'

# Image paths
image_paths = scipy.io.loadmat(meta_dir+'/image-path.mat')
image_paths = [im[0][0] for im in image_paths['imagepath']]

# Attribute labels
with open(meta_dir+'/meta-data-bin.csv','r') as f:
    labels = f.read().strip().split('\n')
labels = [line.strip().split(',')[1:] for line in labels] # ignore image ID
header, labels = np.array(labels[0]), labels[1:]
labels = np.array([map(int, line) for line in labels])
material_idx = [idx for idx, label in enumerate(header) if 'Material' in label]
material_labels = labels[:, material_idx]
material_names = [m.replace('Material.','') for m in header[material_idx]]

images = []
obj_labels = []
attr_labels = []
for idx in range(len(image_paths)):
	path = image_paths[idx]
	top_cat, sub_cat = path.split('/')[0:2]

	# sandals or slippers are full categories
	if top_cat=='Sandals' or top_cat=='Slippers':
		sub_cat = top_cat
	# boots --> ankle, knee_high, mid_calf 
	elif top_cat=='Boots':
		if sub_cat in ['Prewalker Boots', 'Over the Knee']:
			continue
		sub_cat = '%s_%s'%(top_cat, sub_cat)
	# shoes --> Boat, Clogs, Crib, Flats, Heels, Loafers, Oxfords, Sneakers
	elif top_cat=='Shoes':
		if sub_cat in ['Crib Shoes','Firstwalker','Prewalker']:
			continue
		sub_cat = '%s_%s'%(top_cat, sub_cat)

	if material_labels[idx].sum()!=1:
		continue

	sub_cat = sub_cat.replace(' ','_')
	attr_id = material_labels[idx].argmax()
	attr_labels.append(material_names[attr_id])
	obj_labels.append(sub_cat)
	images.append(image_paths[idx])

all_objs = sorted(set(obj_labels))
all_attrs = sorted(set(attr_labels))
attr_labels = [all_attrs.index(attr) for attr in attr_labels]
obj_labels = [all_objs.index(obj) for obj in obj_labels]

pairs = zip(attr_labels, obj_labels)
all_pairs = sorted(set(pairs))

print len(all_objs), len(all_attrs), len(all_pairs)
print len(images)


with open(data_dir + 'all_pairs.txt','w') as f:
    for (attr, obj), count in sorted(collections.Counter(pairs).items()):
        f.write('%s %s %d\n'%(all_attrs[attr], all_objs[obj], count))

pickle.dump([images, all_attrs, all_objs, all_pairs, attr_labels, obj_labels], open(data_dir + 'metadata.pkl','wb'), 2)
