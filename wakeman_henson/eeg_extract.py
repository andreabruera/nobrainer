import os

import mne
from mne.parallel import parallel_func

import pdb

import re

import numpy

import collections

import pickle

# This extracts events from the STI101 channel

def run_events(subject_id, data_path, filter_path):
    print('processing subject: {}'.format(subject_id))
    in_path = os.path.join(data_path, 'sub-{:02}'.format(subject_id), 'ses-meg', 'meg')
    filter_path = os.path.join(filter_path, 'sub-{:02}'.format(subject_id), 'ses-meg', 'meg')
    os.makedirs(filter_path, exist_ok=True)
    for run in range(1, 7):
        run_fname = os.path.join(in_path, 'sub-{:02}_ses-meg_task-facerecognition_run-{:02}_meg.fif'.format(subject_id, run))
        raw = mne.io.read_raw_fif(run_fname)
        mask = 4096 + 256  # mask for excluding high order bits
        events = mne.find_events(raw, stim_channel='STI101',
                                 consecutive='increasing', mask=mask,
                                 mask_type='not_and', min_duration=0.003)

        print('  S {} - R {}'.format(subject_id, run))

        fname_events = os.path.join(filter_path, 'run_{:02}-eve.fif'.format(run))
        mne.write_events(fname_events, events)

# This corrects the names for the wrongly-labeled EEG channels in Wakeman & Henson

def correct_names(raw_eeg):

    raw_eeg.set_channel_types({
        'EEG061': 'eog',
        'EEG062': 'eog',
        'EEG063': 'ecg',
        'EEG064': 'misc'
    })  # EEG064 free floating el.
    raw_eeg.rename_channels({
        'EEG061': 'EOG061',
        'EEG062': 'EOG062',
        'EEG063': 'ECG063'
    })
    
    return raw_eeg

def run_filter(subject_id, data_path, filter_path):
    print('processing subject: {}'.format(subject_id))
    in_path = os.path.join(data_path, 'sub-{:02}'.format(subject_id), 'ses-meg', 'meg')
    filter_path = os.path.join(filter_path, 'sub-{:02}'.format(subject_id), 'ses-meg', 'meg')
    os.makedirs(filter_path, exist_ok=True)
    for run in range(1, 7):
        run_in = os.path.join(in_path, 'sub-{:02}_ses-meg_task-facerecognition_run-{:02}_meg.fif'.format(subject_id, run))
        #run_out = os.path.join(in_path, 'run-{:02}_meg_filter_1_120.fif'.format(run))
        run_out = os.path.join(filter_path, 'run-{:02}_meg_filter_1_70.fif'.format(run))
        raw = mne.io.read_raw_fif(run_in, preload=True, verbose='error')
        raw = correct_names(raw)

        # Band-pass the data channels (MEG and EEG)
        raw.filter(
            #None, 120, l_trans_bandwidth='auto', h_trans_bandwidth='auto',
            None, 70, l_trans_bandwidth='auto', h_trans_bandwidth='auto',
            filter_length='auto', phase='zero', fir_window='hamming',
            fir_design='firwin')
        # High-pass EOG to get reasonable thresholds in autoreject
        picks_eog = mne.pick_types(raw.info, meg=False, eog=True)
        raw.filter(
            1., None, picks=picks_eog, l_trans_bandwidth='auto',
            filter_length='auto', phase='zero', fir_window='hann',
            fir_design='firwin')
        raw.save(run_out, overwrite=True)

# A function to extract epochs for one subject

def run_epochs(subject_id, data_path, filter_path):
    print('processing subject: {}'.format(subject_id))
    in_path = os.path.join(data_path, 'sub-{:02}'.format(subject_id), 'ses-meg', 'meg')
    filter_path = os.path.join(filter_path, 'sub-{:02}'.format(subject_id), 'ses-meg', 'meg')

    for run in range(1, 7):

        events_list = list()

        '''
        bads = list()

        bad_name = op.join('bads', mapping, 'run_%02d_raw_tr.fif_bad' % run)
        if os.path.exists(bad_name):
            with open(bad_name) as f:
                for line in f:
                    bads.append(line.strip())
        '''

        #run_in = os.path.join(in_path, 'run-{:02}_meg_filter_1_120.fif'.format(run))
        run_in = os.path.join(filter_path, 'run-{:02}_meg_filter_1_70.fif'.format(run))
        events_with_id = os.path.join(in_path, 'sub-{:02}_ses-meg_task-facerecognition_run-{:02}_events.tsv'.format(subject_id, run))

        with open(events_with_id) as id_events: 
            events_with_id = [l.strip().split('\t') for l in id_events.readlines()][1:]

        raw = mne.io.read_raw_fif(run_in, preload=True)

        delay = int(round(0.0345 * raw.info['sfreq']))
        events = mne.read_events(os.path.join(filter_path, 'run_{:02}-eve.fif'.format(run)))

        try:
            assert len(events) == len(events_with_id)
        except AssertionError:
            placeholder = events.copy() if len(events) > len(events_with_id) else events_with_id.copy()
            other = events.copy() if len(events) < len(events_with_id) else events_with_id.copy()
            marker = True if len(events) > len(events_with_id) else False
            repetitions = abs(len(events) - len(events_with_id))

            for rep in range(repetitions):
                for index, ev in enumerate(placeholder):
                    value_one = ev[2] if marker else ev[4]
                    value_two = other[index][4] if marker else other[index][2]

                    if int(value_one) != int(value_two):
                        wrong_index = index
                        break

                if marker:
                    placeholder = numpy.delete(placeholder, wrong_index, axis=0)
                else:
                    del placeholder[wrong_index]

            #print([len(placeholder), len(other)])
            for p, o in zip(placeholder, other):
                value_one = p[2] if marker else p[4]
                value_two = o[4] if marker else o[2]
                try:
                    assert int(value_one) == int(value_two)
                except AssertionError:
                    print([value_one, value_two])
            
            if marker:
                events = placeholder.copy()
            else:
                events_with_id = placeholder.copy()
            del placeholder


        # Collapsing all famous events
        for e_one, e_two in zip(events, events_with_id):
            if e_one[2] == 5 or e_one[2] == 6 or e_one[2] == 7:
               e_one[2] = 5
               e_two[-2] = 5

        events[:, 0] = events[:, 0] + delay
        #events_list.append(events)

        '''
        raw.info['bads'] = bads
        raw.interpolate_bads()
        '''

        #raw_list.append(raw)

        #raw, events = mne.concatenate_raws(raw_list, events_list=events_list)
        #raw, events = mne.concatenate_raws(raw, events_list=events)
        raw.set_eeg_reference(projection=True)
        #del raw_list

        picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=True,
                               eog=True, exclude=())

        # Epoch the data
        print('  Epoching')
        epochs = mne.Epochs(raw, events, events_id, tmin, tmax, proj=True,
                            picks=picks, baseline=baseline, preload=False,
                            #decim=6, reject=None, reject_tmax=reject_tmax)
                            decim=4, reject=None, reject_tmax=reject_tmax)

        '''
        # ICA
        ica_name = op.join(meg_dir, subject, 'run_concat-ica.fif')
        ica_out_name = op.join(meg_dir, subject,
                               'run_concat_highpass-%sHz-ica.fif' % (l_freq,))
        print('  Using ICA')
        ica = read_ica(ica_name)
        ica.exclude = []

        filter_label = '-tsss_%d' % tsss if tsss else '_highpass-%sHz' % l_freq
        ecg_epochs = create_ecg_epochs(raw, tmin=-.3, tmax=.3, preload=False)
        eog_epochs = create_eog_epochs(raw, tmin=-.5, tmax=.5, preload=False)
        del raw

        n_max_ecg = 3  # use max 3 components
        ecg_epochs.decimate(5)
        ecg_epochs.load_data()
        ecg_epochs.apply_baseline((None, None))
        ecg_inds, scores_ecg = ica.find_bads_ecg(ecg_epochs, method='ctps',
                                                 threshold=0.8)
        print('    Found %d ECG indices' % (len(ecg_inds),))
        ica.exclude.extend(ecg_inds[:n_max_ecg])
        ecg_epochs.average().save(op.join(data_path, '%s%s-ecg-ave.fif'
                                          % (subject, filter_label)))
        np.save(op.join(data_path, '%s%s-ecg-scores.npy'
                        % (subject, filter_label)), scores_ecg)
        del ecg_epochs

        n_max_eog = 3  # use max 2 components
        eog_epochs.decimate(5)
        eog_epochs.load_data()
        eog_epochs.apply_baseline((None, None))
        eog_inds, scores_eog = ica.find_bads_eog(eog_epochs)
        print('    Found %d EOG indices' % (len(eog_inds),))
        ica.exclude.extend(eog_inds[:n_max_eog])
        eog_epochs.average().save(op.join(data_path, '%s%s-eog-ave.fif'
                                          % (subject, filter_label)))
        np.save(op.join(data_path, '%s%s-eog-scores.npy'
                        % (subject, filter_label)), scores_eog)
        del eog_epochs

        ica.save(ica_out_name)
        epochs.load_data()
        ica.apply(epochs)

        print('  Getting rejection thresholds')
        reject = get_rejection_threshold(epochs.copy().crop(None, reject_tmax),
                                         random_state=random_state)
        epochs.drop_bad(reject=reject)
        print('  Dropped %0.1f%% of epochs' % (epochs.drop_log_stats(),))
        '''

        #epochs.save(os.path.join(in_path, 'run-{:02}_filter_1_120-famous-epo.fif'.format(run)))
        epochs.save(os.path.join(filter_path, 'run-{:02}_filter_1_70-famous-epo.fif'.format(run)), overwrite=True)
        with open(os.path.join(filter_path, 'run-{:02}-famous-epo.eve'.format(run)), 'w') as o:
            for l in events_with_id:
                if int(l[-2]) == 5:
                    o.write('{}\n'.format(l[-1]))


#data_folder = '/import/cogsci/andrea/github/fame/ds000117-download'
#data_folder = '/homes/ab342/ds000117-download/'
data_folder = '/mnt/z/ds000117-completo'
filter_folder = '/mnt/z/ds000117-eeg_filter'
out_path = '/mnt/z/eeg_images_new'

os.makedirs(filter_folder, exist_ok=True)
os.makedirs(out_path, exist_ok=True)

###############################################################################
# We define the events and the onset and offset of the epochs

events_id = {
    'famous' : 5,
    #'face/famous/first': 5,
    #'face/famous/immediate': 6,
    #'face/famous/long': 7,
    #'face/unfamiliar/first': 13,
    #'face/unfamiliar/immediate': 14,
    #'face/unfamiliar/long': 15,
    #'scrambled/first': 17,
    #'scrambled/immediate': 18,
    #'scrambled/long': 19,
}

baseline = (None, 0)
tmin = -0.2
tmax = 2.9  # min duration between onsets: (400 fix + 800 stim + 1700 ISI) ms
#reject_tmax = 0.8  # duration we really care about
reject_tmax = 1.3  # duration we really care about
random_state = 42

with open('wiki_stimuli.txt') as known_file:
    known_ids_all = [l.strip().split('\t') for l in known_file.readlines()]
known_ids = [l[0] for l in known_ids_all if len(l) > 1]


# Extracting events

#parallel, run_func, _ = parallel_func(run_events, n_jobs=4)
#parallel(run_func(subject_id, data_folder, filter_folder) for subject_id in range(1, 17))
#for s in range(1, 17):
    #run_events(s, data_folder, filter_folder)

# Filtering the data
#for i in range(1, 17):
    #run_filter(i, data_folder, filter_folder)

# Epoching the data
#wrong_subs = [1]
#for i in range(1, 17):
    #run_epochs(i, data_folder, filter_folder)

for s in range(1, 17):
    out_dict = collections.defaultdict(list)
    #for s in [int(1)]:
    filtered_path = os.path.join(filter_folder, 'sub-{:02}'.format(s), 'ses-meg', 'meg')
    for run in range(1, 7):
        #epochs = mne.read_epochs(os.path.join(in_path, 'run-06_filter_1_120-famous-epo.fif'))
        #epochs = mne.read_epochs(os.path.join(in_path, 'run-06_filter_1_70-famous-epo.fif'))
        epochs = mne.read_epochs(os.path.join(filtered_path, 'run-{:02}_filter_1_70-famous-epo.fif'.format(run)))
        #with open(os.path.join(in_path, 'run-06-famous-epo.eve')) as o:
        with open(os.path.join(filtered_path, 'run-{:02}-famous-epo.eve'.format(run))) as o:
            ids = [l.strip() for l in o.readlines()]
        data = epochs.get_data(picks='eeg')
        scaler = mne.decoding.Scaler(epochs.info)
        scaled_data = scaler.fit_transform(data)
        vectorizer = mne.decoding.Vectorizer()
        vector_data = vectorizer.fit_transform(data)


        for identity, vector in zip(ids, vector_data):
            identity = re.sub('.+/', '', identity)
            if identity in known_ids:
                out_dict[re.sub('.+/|\.bmp', '', identity)].append(vector)

    with open(os.path.join(out_path, 'sub-{:02}_eeg_vecs.pkl'.format(s)), 'wb') as o:
        pickle.dump(out_dict, o)
