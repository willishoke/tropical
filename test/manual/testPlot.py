import sys
import numpy as np
import matplotlib.pyplot as plt

fname = sys.argv[1]
testname = sys.argv[2]

# ignore final value
data = np.genfromtxt(fname, delimiter=',')[:-1] 

plt.figure(figsize=(12,3))
plt.plot(data, linewidth=.5)
plt.title(testname + " output waveform: " + str(len(data)) + " samples")
plt.show()
