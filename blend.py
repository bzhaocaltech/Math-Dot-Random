import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('results', nargs='+')

args = parser.parse_args()

blend_data = np.array([np.loadtxt(res + 'blend.dta') for res in args.results]).T
probe_y = np.load('um/probe.npy')[:, 3]
c, residuals, rank, s = np.linalg.lstsq(blend_data, probe_y, rcond=None)

print('c_vals: {}'.format(c))

qual_data = np.array([np.loadtxt(res + 'results.dta') for res in args.results])
blend_results = np.zeros(shape=(2749898,))
for i, c_val in enumerate(c):
    blend_results += qual_data[i] * c_val

print('blend results: {}'.format(blend_results))

with open('blend.dta', 'w') as blend_file:
    for v in blend_results:
        blend_file.write('{}\n'.format(v))
