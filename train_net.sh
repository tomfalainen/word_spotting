#!/bin/sh
th train.lua -dataset washington_fold_1_synth -id washington_triplet_fold_1
th train_we.lua -dataset washington_fold_1_synth_dct3 -id washington_dct3_fold_1 -weights checkpoints/washington_triplet_fold_1_iter_50000.t7
th train_we.lua -dataset washington_fold_1_synth_ngram -id washington_ngram_fold_1 -weights checkpoints/washington_triplet_fold_1_iter_50000.t7
th train_we.lua -dataset washington_fold_1_synth_semantic -id washington_semantic_fold_1 -weights checkpoints/washington_triplet_fold_1_iter_50000.t7
th train_we.lua -dataset washington_fold_1_synth_phoc -id washington_phoc_fold_1 -weights checkpoints/washington_triplet_fold_1_iter_50000.t7
