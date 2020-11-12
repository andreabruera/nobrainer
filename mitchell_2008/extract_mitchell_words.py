import scipy.io
import os
import pdb

words_and_cats = []
sample = scipy.io.loadmat('data-science-P1.mat')

for s in sample['info'][0]:
    info = (s[2][0], s[0][0])
    if info not in words_and_cats:
        words_and_cats.append(info)

with open('mitchell_words_and_cats.txt', 'w') as o:
    for i in words_and_cats:
        o.write('{}\t{}\n'.format(i[0], i[1]))
