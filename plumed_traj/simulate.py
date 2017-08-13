#!/bin/env python

from load_sim import create_simulation
from mpi4py import MPI
import numpy as np
from simtk.unit import *

import sys,os
from plumed import *


sim_temp = 300
beta = 0.0083144621 * 300


base_dir = "./"
rank = 0
sim_obj = create_simulation(base_dir, rank ,globals()["plumed_%d"%rank]) 
for step in range(2000):
    print(step)
    #2fs *5000 =10ps
    sim_obj.step(5000)
    #write the chckpt
    with open("checkpt.chk",'wb') as f:
        f.write(sim_obj.context.createCheckpoint())
