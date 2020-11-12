import collections
import nilearn
import os
import math
import re
import nilearn.input_data
import numpy
import pickle

from tqdm import tqdm

def get_events(subject, run, folder, dataset='wakeman_henson'):

    run = int(run)
    subject = int(subject)
    identities_list = [l.strip().split('\t') for l in open('resources/wakeman_henson_stimuli.txt').readlines()]
    identities_dict = {re.sub('\..+', '', k[0]) : k[1] for k in identities_list if len(k) > 2}

    timepoints = collections.defaultdict(list)
    individual_timepoints = collections.defaultdict(str)

    with open(os.path.join(folder, 'sub_{:02}'.format(subject), 'func', 'run_{:02}'.format(run), 'run_{:02}.events'.format(run))) as f:
        f = [l.split() for l in f.readlines()][1:]
        for l in f:
            time = int(math.ceil(float(l[0])))
            if subject == 10 and run == 9 and time > 330:
                pass
            else:
                if dataset == 'ltm_fame':
                    condition = l[2]
                    identity = 'unknown'
                else:
                    condition = str(l[3]).lower()
                    identity = re.sub('func/|.bmp', '', str(l[7]))
                    
                if condition.isalpha():
                    timepoints[condition].append(time)
                    if re.search('f.', identity) and identity in identities_dict.keys(): 
                        individual_timepoints[time] = identities_dict[identity]
                else:
                    if 'i999' in l[7]:
                        timepoints['subtraction'].append(time)

    return timepoints, individual_timepoints

def get_wakeman_henson_images(pickling=True, dataset='wakeman_henson', smoothed=False):

    masker = nilearn.input_data.NiftiMasker(mask_strategy='template', detrend = True, high_pass = 0.005, t_r = 2)
    in_folder = os.path.join('data', 'neuro_data', 'mmms_fame')
    out_folder = 'data/wakeman_henson_updated_pickles'
    os.makedirs(out_folder, exist_ok=True)

    if smoothed:
        four_d_filename = '4D_smooth'
    else:
        four_d_filename = '4D_norm'

    for subject in range(1, 17):

        print('Currently collecting fMRI brain images for subject {}...'.format(subject))
        sub_images_individuals = collections.defaultdict(list)

        for run in tqdm(range(1, 10)):

            subtraction = []

            timepoints, individual_timepoints = get_events(subject, run, in_folder)
            run_img = nilearn.image.load_img(os.path.join(in_folder, 'sub_{:02}'.format(subject), 'func',  'run_{:02}'.format(run), '{}_run_{:02}.nii'.format(four_d_filename, run)))
            masked_img = masker.fit_transform(run_img)

            for condition, times in timepoints.items():
                for t in times:

                    if t in individual_timepoints.keys():

                        starting_time = min([k for k in range(len(masked_img)) if k * 2 >= t])  ### 2 because 2 is the t_r for wakeman & henson
                        average_img = numpy.average(masked_img[starting_time+2:starting_time+5], axis =0) # Range of seconds considered after stimulus onset: 4-6-8
                        sub_images_individuals[individual_timepoints[t]].append(average_img)

                    elif condition == 'subtraction':

                        starting_time = min([k for k in range(len(masked_img)) if k * 2 >= t])  ### 2 because 2 is the t_r for wakeman & henson
                        average_img = numpy.average(masked_img[starting_time+2:starting_time+7], axis =0) # Range of seconds considered after stimulus onset: much larger because fixation cross is much longer (from 4 to 12 seconds after fixation)
                        subtraction.append(average_img)

        # Dump to pickle the brain images for the individuals

        if pickling:
            if smoothed:
                pickle_path = os.path.join(out_folder, 'fmri_sub_{:02}_smoothed.pkl'.format(subject))
            else:
                pickle_path = os.path.join(out_folder, 'fmri_sub_{:02}.pkl'.format(subject))

            with open(pickle_path, 'wb') as o:
                pickle.dump(sub_images_individuals, o)

get_wakeman_henson_images(smoothed=True)
