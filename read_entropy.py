import numpy as np
import glob, os

for f in glob.glob('entropy_*.out'):
    x = np.loadtxt(f)
    print(f, np.mean(x), np.median(x), np.amin(x), np.amax(x))

