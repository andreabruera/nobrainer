import pickle
import collections

import scipy.io

mapping_dict = {'cow' : 'Cattle', 'corn' : 'Maize', 'pants' : 'Trousers'}

for i in range(1, 10):
    current_pickle = collections.defaultdict(list)
    current_file = scipy.io.loadmat('data-science-P{}.mat'.format(i))
    trials = [k[2][0].capitalize() if k[2][0] not in mapping_dict.keys() else mapping_dict[k[2][0]] for k in current_file['info'][0]]
    data = [d[0][0] for d in current_file['data']]
    for k, v in zip(trials, data):
        current_pickle[k].append(v)
    with open('mitchell_pickles/sub_{:02}.pkl'.format(i), 'wb') as o:
        pickle.dump(current_pickle, o)
