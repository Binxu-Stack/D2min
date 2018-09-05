#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A script to solve D2min field.
"""
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

# Python template written by Bin Xu (xubinrun@gmail.com)
# Licensed under GPL License

# sys library is often used

import argparse
#import os
#import re
import sys
import numpy as np
import time
from numpy.linalg import inv
#import pandas as pd
#import matplotlib.pyplot as plt
#import doctest
#It's a good habit to write testing code.

# mysql connection
#from sqlalchemy import create_engine
#engine = create_engine('mysql://research@localhost/research', echo=False)
# df.to_sql(table_name,engine,if_exists='replace')
#with engine.connect() as con:
#    data = pd.read_sql_table(table_name, con,index_col = "index")

def get_d2min(bonds0, bonds):
    """ Calculate the d2min of two 2d arrays
    Args:
        bonds0: N*d double array, reference bonds vector
        bonds:  N*d double array, current bonds vector

    Returns:
        (d2min, J, eta_s), eta_s is the von-Mises strain

    Notes:
        V = bonds0.T * bonds0, 3*N N*3 = 3*3
        W = bonds0.T * bonds 
        J = V^-1 * W
        d2min = |bonds0 - J*bonds|
        eta = 0.5*(J*J.T-I), Lagrangian strain matrix
        eta_m = 1.0/3.0*trace(eta), dilational component
        eta_s = \sqrt{0.5*trace(eta-eta_m*I)}, von-Mises strain
    Examples:
        b0 = [[1,0], [0,1]]
        b = [[1,0.1], [0.1,1]]
        d2min, J, eta_s = get_d2min(b0,b)

        
    Reference:
        Falk, M. L., & Langer, J. S. (1998)
        http://li.mit.edu/Archive/Graphics/A/
    """
    # convert to numpy matrix
    b0 = np.mat(bonds0)
    b = np.mat(bonds)

    # get the dimension of bonds
    dimension = b0.shape[1]

    # get V and W
    V = b0.transpose() * b0
    W = b0.transpose() * b

    # calculate J
    #print("V:",V)
    #print("b0:",b0)
    #print("b:",b)
    J = inv(V) * W

    # non-affine part
    non_affine = b0*J - b

    # d2min
    d2min = np.sum(np.square(non_affine))

    # get Lagrangian strain matrix
    eta = 0.5 * (J * J.transpose() - np.eye(dimension))

    # get von-Mises strain
    if dimension == 0:
        print("dimension:",dimension)
        print("b0:",b0)
        print("b:",b)
        sys.exit(1)
    eta_m = 1.0/np.double(dimension) * np.trace(eta)
    tmp = eta - eta_m * np.eye(dimension)
    eta_s = np.sqrt(0.5*np.trace(tmp*tmp))
    #print(J)
    #print(d2min)
    #print(eta_s)
    #sys.exit()
    return (d2min, J, eta_s)


from ase.io import read
from ase.neighborlist import neighbor_list
from ase.neighborlist import mic
def get_d2min_config(config0, config, config_format, cutoff, dimension=2):
    """ Get the d2min field of a configuration refer to one configuration.
    Args:
        config0: str, reference configuration, should be supported by ASE.io
        config: str, current configuration
        config_format: str, format of configuration, e.g. 'lammps-dump'
        cutoff: double, cutoff to build neighbor list
    Return:
        Natom array of tuple (d2min, J, eta_s) sorted in the order of atom id.
    """

    # read atoms from configurations
    atoms0 = read(config0, format=config_format)
    atoms = read(config, format=config_format)
    if dimension == 2:
        atoms0.set_pbc([True,True,False])
        atoms.set_pbc([True,True,False])
    elif dimension == 3:
        atoms0.set_pbc(True)
        atoms.set_pbc(True)

    #natoms = len(atoms0.positions)

    d2mins = []
    eta_ss = []
    Js = []

    # build neighbour list on reference configuration using cutoff
    # neighbour list is sorted according to i
    ilist,jlist,Dlist0 = neighbor_list('ijD', atoms0, cutoff)
    nbonds = len(ilist)
    bonds0 = []
    bonds = []
    current_id = 0
    cell = atoms.get_cell()
    #neighbors = []
    #neighbors.append(atoms0.positions[0][:2])
    for i, j, D0 in zip(ilist,jlist, Dlist0):
        if i == current_id:
            #neighbors.append(atoms0.positions[j][:2])
            #print("i:",i)
            #print("j:",j)
            #print("D0:",D0)
            bonds0.append(D0[:dimension])
            dr = atoms.positions[j] - atoms.positions[i]
            dr = mic(dr, cell)
            #distance = np.sqrt(sum(dr*dr))
            #print("distance:",distance)
            bonds.append(dr[:dimension])
        else:
            d2min, J, eta_s = get_d2min(bonds0,bonds)
            d2mins.append(d2min)
            eta_ss.append(eta_s)
            Js.append(J)
            bonds0.clear()
            bonds.clear()
            bonds0.append(D0[:dimension])
            dr = atoms.positions[j] - atoms.positions[i]
            dr = mic(dr, cell)
            bonds.append(dr[:dimension])
            current_id = i
    # for the last atom
    d2min, J, eta_s = get_d2min(bonds0,bonds)
    d2mins.append(d2min)
    eta_ss.append(eta_s)
    Js.append(J)

    return (d2mins, Js, eta_ss)

    

def test():
    b0 = [[1,0], [0,1]]
    b = [[1,0.1], [0.1,1]]
    d2min, J, eta_s = get_d2min(b0,b)
    print(d2min)
    print(J)
    print(eta_s)
    assert(d2min==0)
    assert(eta_s==0.1)




# main process



def main():
    # Time start
    start_ticks = time.time()

    # Parse arguments
    parser = argparse.ArgumentParser(description=__doc__,
            epilog= """Script to calculate d2min and von-Mises strain for atoms in configurations.
    output to d2min.aux and etas.aux.
                    """,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-nt","--notime", help="Don't dispaly time usage.", action="store_true", default=False)
    #parser.add_argument("-k","--key", help="Define the key column.", type=int, default=1, metavar='col_num', nargs='+')
    parser.add_argument("-d","--dimension", help="Define the dimension of configuration.", type=int, default=2)
    parser.add_argument("-c","--cutoff", help="Define the cutoff to build neighbour list.", type=float, default=2.5)
    parser.add_argument("-f","--format", help="Define the format of configuration.", default='lammps-dump')
    #parser.add_argument("-k","--key", help="Define the key column.", type=int, default=1, metavar='col_num', nargs='+')
    parser.add_argument("infiles", help="Define the input file.", nargs=2)
    #parser.add_argument("-o","--outfile", help="Define the output file.", type=argparse.FileType('w'), default=sys.stdout)
    args = parser.parse_args()

    # Your main process
    #test()
    d2mins, Js, eta_ss = get_d2min_config(args.infiles[0], args.infiles[1], args.format, args.cutoff, args.dimension)

    with open('d2min.aux','w') as out:
        for d2min in d2mins:
            out.write("%16f\n"%d2min)

    with open('etas.aux','w') as out:
        for eta in eta_ss:
            out.write("%16f\n"%eta)


    
    # Time end
    end_tickes = time.time()
    # Output time usage
    if not args.notime:
        print("Time usage: %f s" % (end_tickes-start_ticks))
    return 0

if __name__ == "__main__":
    sys.exit(main())
