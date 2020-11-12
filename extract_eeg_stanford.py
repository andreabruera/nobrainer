from scipy.io import loadmat
import os
import collections
import pickle

from tqdm import tqdm

with open('/mnt/z/Dataset EEG/Object Category EEG dataset/eeg_stanford_stimuli_ids.txt') as i: 
    #present_ids = [l.strip().split('\t') for l in i.readlines() if len(l.strip().split('\t')) > 1]
    present_ids = [l.strip().split('\t') for l in i.readlines()]
mapping = {l[0] : l[1] for l in present_ids}
    
for s in tqdm(range(1, 11)):
    to_be_pickled = collections.defaultdict(list)
    s_name = 'S{}_'.format(s)

    for root, direct, files in os.walk('/mnt/z/Dataset EEG/Object Category EEG dataset'):
        for f in files:
            if s_name in f:

                ### Loading information about the data
                a = loadmat(os.path.join(root, f))
                ids = [k[0] for k in a['exemplarLabels']]
                vecs = a['xEpoched']
                shape = vecs.shape
                channels = 125 # We could use shape[0] if we wanted to use all 129 electrodes, but we want to follow the original paper and only use the 125 first electrodes
                timesteps = shape[1]
                exemplars = shape[2]
                
                ### Reorganizing the data structure so as to obtain a dictionary, where for each individual there are 72 brain images, each divided into 129 channels
                for e in range(exemplars):
                    current_exemplar = []
                    for c in range(channels):
                        ex_to_be_pickled = []
                        for t in range(timesteps):
                            ex_to_be_pickled.append(vecs[c][t][e])
                        current_exemplar.append(ex_to_be_pickled)
                    to_be_pickled[ids[e]].append(current_exemplar)

    general_out_folder = '/mnt/z/Dataset EEG/Object Category EEG dataset/vectors'.format(s)

    ### Writing to txt file the concatenated vector
    out_folder = os.path.join(general_out_folder, 'concatenated_vectors/sub_{:02}'.format(s))
    os.makedirs(out_folder, exist_ok=True)
    print('Writing to file...')
    for pres_id in present_ids:
        out_name = str(pres_id[1])
        with open(os.path.join(out_folder, '{}.vec'.format(out_name)), 'w') as o:
            key = int(pres_id[0])
            for v in to_be_pickled[key]:
                for channel in v:
                    for value in channel:
                        o.write('{}\t'.format(value))
                o.write('\n')

    '''
    ### Dumping to pickle the original structure vector
    out_folder = os.path.join(general_out_folder, 'eeg_stanford_pickles/sub_{:02}'.format(s))
    os.makedirs(out_folder, exist_ok=True)
    final_pickle = {mapping[str(k)] : v for k, v in to_be_pickled.items()}
    print('Writing to pickle...')
    with open(os.path.join(out_folder, 'sub_{:02}.pkl'.format(s)), 'wb') as o:
        pickle.dump(final_pickle, o)
    '''
