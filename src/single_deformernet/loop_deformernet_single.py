import sys
import os
import numpy as np
import timeit
from itertools import product

start_time = timeit.default_timer()

### Train single

prim_names = ["box"]  # ["box", "cylinder", "hemis"]
stiffnesses = ["5k", "10k"]  # ["1k", "5k", "10k"]
# os.chdir("./")

for (prim_name, stiffness) in list(product(prim_names, stiffnesses)):
    obj_category = f"{prim_name}_{stiffness}Pa"
    os.system(f"python3 process_data_single.py --obj_category {obj_category}")
    os.system(f"python3 single_trainer_all_objects.py")

print(f"DONE! You burned {(timeit.default_timer() - start_time)/3600} trees")
