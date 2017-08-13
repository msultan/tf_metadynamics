import os
import mdtraj
import mdtraj.reporters

# And a few things froms OpenMM
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *

from simtk import unit
import simtk.openmm as mm
from simtk.openmm import app

import mdtraj.testing



def serializeObject(obj,objname):
    objfile = open(objname,'w')
    objfile.write(XmlSerializer.serialize(obj))
    objfile.close()


pdb = app.PDBFile('0.pdb')

# Lets use the amber99sb-ildn forcefield with TIP3P explicit solvent
# and a langevin integrator. This is relatively "standard" OpenMM
# code for setting up a system.
forcefield = app.ForceField('amber99sbildn.xml', 'tip3p.xml')

#basic system
system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.PME, 
    nonbondedCutoff=1.0*unit.nanometers, constraints=app.HBonds, rigidWater=True, 
    ewaldErrorTolerance=0.0005)
#NPT
system.addForce(mm.MonteCarloBarostat(1*unit.atmospheres, 300*unit.kelvin, 25))
#vanilla langevin integrator 
integrator = mm.LangevinIntegrator(300*unit.kelvin, 1.0/unit.picoseconds, 
    2.0*unit.femtoseconds)
integrator.setConstraintTolerance(0.00001)

#set positions
platform = mm.Platform.getPlatformByName('CPU')
simulation = app.Simulation(pdb.topology, system, integrator, platform)
simulation.context.setPositions(pdb.positions)


simulation.step(1)

state=simulation.context.getState(getPositions=True, getVelocities=True,\
    getForces=True,getEnergy=True,getParameters=True,enforcePeriodicBox=True)

serializeObject(state,'state.xml')
serializeObject(system,'system.xml')
serializeObject(integrator,'integrator.xml')
