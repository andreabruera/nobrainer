import nilearn
import numpy
import os
import pickle
import scipy

from nilearn import datasets, image, masking
from scipy import stats
from tqdm import tqdm

out_folder = 'pickles'
os.makedirs(out_folder, exist_ok=True)

### loading mask
mask_file = 'Visual_Semantic_Cognition_ALE_result.nii'
original_mask = nilearn.image.load_img(mask_file)
array_mask = numpy.array(nilearn.image.binarize_img(original_mask).get_fdata(), dtype=bool)
mask = nilearn.image.new_img_like(original_mask, array_mask)

TR = 2.25

entities = dict()
with open('transe_stimuli.txt') as i:
    for l in i:
        line = l.strip().split('\t')
        if len(line) > 1:
            entities['func/{}'.format(line[0])] = line[1]

for s in range(1, 17):
    sub_folder = os.path.join('sub-{:02}'.format(s),'ses-mri', 'func')
    assert os.path.exists(sub_folder)
    sub_data = dict()
    for r in range(1, 10):
        ### fmri
        #file_name = 'sub-{:02}_ses-mri_task-facerecognition_run-{}_desc-preproc_bold.nii.gz'.format(s, r)
        file_name = 'sub-{:02}_ses-mri_task-facerecognition_run-{}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz'.format(s, r)
        file_path = os.path.join(sub_folder, file_name)
        assert os.path.exists(file_path)
        img = nilearn.image.load_img(file_path)
        marker = False
        if not marker:
            resampled_mask = nilearn.image.resample_img(mask, img.affine, target_shape=img.shape[:3])
            marker = True
        masked_img = nilearn.masking.apply_mask(img, resampled_mask)
        print(masked_img.shape)
        ### events
        file_name = 'sub-{:02}_ses-mri_task-facerecognition_run-{:02}_events.txt'.format(s, r)
        file_path = os.path.join(sub_folder, file_name)
        print(file_path)
        assert os.path.exists(file_path)
        with open(file_path) as i:
            counter = 0
            for l in i:
                line = l.strip().split('\t')
                if counter == 0:
                    header = line.copy()
                    run_events = {k : list() for k in header}
                    counter += 1
                    continue
                for h in header:
                    run_events[h].append(line[header.index(h)])
        ### matching things
        starting_point = TR * 0.5
        recorded_time_points = [(TR*i)+starting_point for i in range(img.shape[-1])]
        #entity_onsets = {v : float(k) for k, v in zip(run_events['onset'], run_events['stim_file']) if v in entities.keys()}
        #run_entities = [e for e in entities.keys() if e in entity_onsets.keys()]
        #for e in run_entities:
        for onset, e in zip(run_events['onset'], run_events['stim_file']):
            if e not in entities.keys():
                continue
            ### finding the starting idx
            t_differences = [(t_i, abs(float(onset)-float(t))) for t_i, t in enumerate(recorded_time_points)]
            starting_idx = sorted(t_differences, key=lambda item : item[1])[0][0]
            entity = entities[e]
            #print(entity)
            if entity not in sub_data.keys():
                sub_data[entity] = [masked_img[starting_idx:starting_idx+12, :]]
            else:
                sub_data[entity].append(masked_img[starting_idx:starting_idx+12, :])
    sub_data = {k : [val for val in v if val.shape==(12, 9957)] for k, v in sub_data.items()}
    ### top 500 features
    sub_data_feature_selection = {k : numpy.average(numpy.array(v)[:,4:8, :], axis=1) for k, v in sub_data.items() if len(v)==2}
    corrs = list()
    for voxel in tqdm(range(9957)):
        voxel_data = {k : v[:, voxel] for k, v in sub_data_feature_selection.items()}
        corr = scipy.stats.pearsonr([v[0] for k, v in voxel_data.items()], [v[1] for k, v in voxel_data.items()])[0]
        corrs.append(corr)
    ### selecting 500 features
    five_hundred = [v[0] for v in sorted(enumerate(corrs), key=lambda item : item[1], reverse=True)][:500]

    sub_data = {k : numpy.average(numpy.array(v)[:, :, five_hundred], axis=0) for k, v in sub_data.items()}
    with open(os.path.join(out_folder, 'sub-{:02}.pkl'.format(s)), 'wb') as o:
        pickle.dump(sub_data, o)
