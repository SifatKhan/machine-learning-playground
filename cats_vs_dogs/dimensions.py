from PIL import Image
from os import listdir, makedirs
from os.path import isfile, join, isdir, exists
import pandas as pd
import numpy as np

path = 'train/'
file_list = [path + f for f in listdir(path) if isfile(join(path, f))]
widths = []
heights = []

for f in file_list:
    pic = Image.open(f)
    widths.append(pic.size[0])
    heights.append(pic.size[1])

data = np.vstack((widths, heights)).T
df = pd.DataFrame(data=data, columns=['widths', 'heights'])

pass

