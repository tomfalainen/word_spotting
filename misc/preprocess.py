# -*- coding: utf-8 -*-
"""
@author: tomas
"""
import json
import os
import numpy as np
import misc.data_augmentation as da
import misc.db_utils as dbu

root_dir = 'data/washington/'
fold = 1
db = 'washington'
data = dbu.load_washington(fold, root_dir)

texts = [datum['text'] for datum in data]

vocab, indeces = np.unique(texts, return_index=True)
print "size of vocabulary:", len(vocab)

itow = {i+1:w for i,w in enumerate(vocab)} # a 1-indexed vocab translation table
wtoi = {w:i+1 for i,w in enumerate(vocab)} # inverse table

for datum, text in zip(data, texts):
    datum['label'] = wtoi[text]

db = root_dir + db

# Write washington data to h5 database, needed for evaluation purposes
dbu.write_to_db_full(data, db)

#Write accompanying json data
json_output = {}
json_output['data'] = data
json_output['image_db'] = db

with open('%s_preprocessed.json' % db, 'w') as f:
    json.dump(json_output, f)
    
db = db + '_fold_%d' % fold
piece_size = 250000

#Write data augmented washington to h5 data, for training.
augmented_dir = os.path.join(root_dir, 'augmented/')
augmented_data = da.augment_data(data, augmented_dir)
db += '_synth'
inds = np.random.permutation(len(augmented_data))
augmented_data = [augmented_data[i] for i in inds]
dbu.write_to_db(augmented_data, db, piece_size)

#%%
#Write accompanying json data
json_output = {}
json_output['itow'] = itow
json_output['wtoi'] = wtoi
json_output['data'] = augmented_data
json_output['image_db'] = db
json_output['inds'] = inds.tolist()
json_output['pieces'] = np.ceil(np.array(dbu.lens_by_split(augmented_data)).astype(np.float32) / piece_size).astype(np.int32).tolist()

with open('%s_preprocessed.json' % db, 'w') as f:
    json.dump(json_output, f)

#Write embedding databases
for embedding in ['dct3', 'ngram', 'semantic', 'phoc']:
    we = np.load('embeddings/washington_%s_embeddings.npy' % (embedding))
    we = we[indeces, :]
    wtoe = {w:we[i] for i, w in enumerate(vocab)} #word embedding table
    embedding_db = db + '_' + embedding
    
    # Write embedding data to h5
    dbu.write_to_embedding_db(augmented_data, wtoe, embedding_db, piece_size)
    
    #Write accompanying json data
    json_output = {}
    json_output['itow'] = itow
    json_output['wtoi'] = wtoi
    json_output['data'] = data
    json_output['image_db'] = db
    json_output['embedding_db'] = embedding_db
    json_output['pieces'] = np.ceil(np.array(dbu.lens_by_split(augmented_data)).astype(np.float32) / piece_size).astype(np.int32).tolist()
    
    with open('%s_preprocessed.json' % embedding_db, 'w') as f:
        json.dump(json_output, f)
