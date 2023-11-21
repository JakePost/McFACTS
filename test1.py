from cgi import print_arguments
import numpy as np
import math
import matplotlib.pyplot as plt
import itertools
import scipy.interpolate

import sys
import argparse

from inputs import ReadInputs

from setup import setupdiskblackholes
from physics.migration.type1 import type1
from physics.accretion.eddington import changebhmass
from physics.accretion.torque import changebh
from physics.feedback.hankla21 import feedback_hankla21
#from physics.dynamics import wang22
from physics.binary.formation import hillsphere
from physics.binary.formation import add_new_binary
#from physics.binary.formation import secunda20
from physics.binary.evolve import evolve
from physics.binary.harden import baruteau11
from physics.binary.merge import tichy08
from physics.binary.merge import chieff
from physics.binary.merge import tgw
#from tests import tests
from outputs import mergerfile


verbose=False
n_bins_max = 1000
n_bins_max_out = 100
binary_field_names="R1 R2 M1 M2 a1 a2 theta1 theta2 sep com t_gw merger_flag t_mgr  gen_1 gen_2  bin_ang_mom"
merger_field_names=' '.join(mergerfile.names_rec)


parser = argparse.ArgumentParser()
parser.add_argument("--use-ini",help="Filename of configuration file", default=None)
parser.add_argument("--fname-output-mergers",default="output_mergers.dat",help="output merger file (if any)")
parser.add_argument("--fname-snapshots-bh",default="output_bh_[single|binary]_$(index).dat",help="output of BH index file ")
parser.add_argument("--no-snapshots", action='store_true')
parser.add_argument("--verbose",action='store_true')
opts=  parser.parse_args()
verbose=opts.verbose

def main():
    """
    """

    #1. Test a merger by calling modules
    # print("Test merger")
    # mass_1 = 10.0
    # mass_2 = 15.0
    # spin_1 = 0.1
    # spin_2 = 0.7
    # angle_1 = 1.80
    # angle2 = 0.7
    # bin_ang_mom = 1.0
    # outmass = tichy08.merged_mass(mass_1, mass_2, spin_1, spin_2)
    # outspin = tichy08.merged_spin(mass_1, mass_2, spin_1, spin_2, bin_ang_mom)
    # out_chi = chieff.chi_effective(mass_1, mass_2, spin_1, spin_2, angle_1, angle2, bin_ang_mom)
    # print(outmass,outspin,out_chi)
    #Output should always be constant: 23.560384 0.8402299374639024 0.31214563487176167
   #    test_merger=tests.test_merger()

    # Setting up automated input parameters
    # see ReadInputs.py for documentation of variable names/types/etc.

    fname = 'inputs/model_choice.txt'
    if opts.use_ini:
        fname = opts.use_ini
    mass_smbh, trap_radius, disk_outer_radius, alpha, n_bh, mode_mbh_init, max_initial_bh_mass, \
         mbh_powerlaw_index, mu_spin_distribution, sigma_spin_distribution, \
             spin_torque_condition, frac_Eddington_ratio, max_initial_eccentricity, \
                 timestep, number_of_timesteps, disk_model_radius_array, disk_inner_radius,\
                     disk_outer_radius, surface_density_array, aspect_ratio_array, retro, feedback, capture_time, outer_capture_radius\
                     = ReadInputs.ReadInputs_ini(fname)

    # create surface density & aspect ratio functions from input arrays
    surf_dens_func_log = scipy.interpolate.UnivariateSpline(disk_model_radius_array, np.log(surface_density_array))
    surf_dens_func = lambda x, f=surf_dens_func_log: np.exp(f(x))

    aspect_ratio_func_log = scipy.interpolate.UnivariateSpline(disk_model_radius_array, np.log(aspect_ratio_array))
    aspect_ratio_func = lambda x, f=aspect_ratio_func_log: np.exp(f(x))

    # mass_smbh, trap_radius, n_bh, mode_mbh_init, max_initial_bh_mass, \
    #      mbh_powerlaw_index, mu_spin_distribution, sigma_spin_distribution, \
    #          spin_torque_condition, frac_Eddington_ratio, max_initial_eccentricity, \
    #              timestep, number_of_timesteps, disk_model_radius_array, disk_inner_radius,\
    #                  disk_outer_radius, surface_density_array, aspect_ratio_array \
    #                     = ReadInputs.ReadInputs()
    
    
    print(" Number of BHs ", n_bh)

    print("Generate initial BH parameter arrays")
    bh_initial_locations = setupdiskblackholes.setup_disk_blackholes_location(n_bh, disk_outer_radius)
    bh_initial_masses = setupdiskblackholes.setup_disk_blackholes_masses(n_bh, mode_mbh_init, max_initial_bh_mass, mbh_powerlaw_index)
    bh_initial_spins = setupdiskblackholes.setup_disk_blackholes_spins(n_bh, mu_spin_distribution, sigma_spin_distribution)
    bh_initial_spin_angles = setupdiskblackholes.setup_disk_blackholes_spin_angles(n_bh, bh_initial_spins)
    bh_initial_orb_ang_mom = setupdiskblackholes.setup_disk_blackholes_orb_ang_mom(n_bh)
    bh_initial_orb_ecc = setupdiskblackholes.setup_disk_blackholes_eccentricity_thermal(n_bh)
    #print("orb ecc",bh_initial_orb_ecc)
    #bh_initial_generations = np.ones((integer_nbh,),dtype=int)  
    bh_initial_generations = np.ones((n_bh,),dtype=int)

    #3.a Test migration of prograde BH
    # Disk surface density (in kg/m^2) is a function of radius, where radius is in r_g
    disk_surface_density = surf_dens_func
    # and disk aspect ratio is also a function of radius, where radius is in r_g
    disk_aspect_ratio = aspect_ratio_func
    #Housekeeping: Set up time
    initial_time = 0.0
    final_time = timestep*number_of_timesteps
    print("Migrate BH in disk")
    #Find prograde BH orbiters. Identify BH with orb. ang mom =+1
    bh_orb_ang_mom_indices = np.array(bh_initial_orb_ang_mom)
    prograde_orb_ang_mom_indices = np.where(bh_orb_ang_mom_indices == 1)
    #retrograde_orb_ang_mom_indices = np.where(bh_orb_ang_mom_indices == -1)
    prograde_bh_locations = bh_initial_locations[prograde_orb_ang_mom_indices]
    sorted_prograde_bh_locations = np.sort(prograde_bh_locations)
    print("Sorted prograde BH locations:", len(sorted_prograde_bh_locations), len(prograde_bh_locations))
    print(sorted_prograde_bh_locations)

    #b. Test accretion onto prograde BH
    # Housekeeping: Fractional rate of mass growth per year at 
    # the Eddington rate(2.3e-8/yr)
    mass_growth_Edd_rate = 2.3e-8
    #Use masses of prograde BH only
    prograde_bh_masses = bh_initial_masses[prograde_orb_ang_mom_indices]
    print("Prograde BH initial masses", len(prograde_bh_masses))
#    print(prograde_bh_masses)

    #c. Test spin change and spin angle torquing
    #Housekeeping: minimum spin angle resolution 
    # (ie less than this value gets fixed to zero) 
    # e.g 0.02 rad=1deg
    spin_minimum_resolution = 0.02
    #Torque prograde orbiting BH only
    print("Prograde BH initial spins")
    prograde_bh_spins = bh_initial_spins[prograde_orb_ang_mom_indices]
#    print(prograde_bh_spins)
    print("Prograde BH initial spin angles")
    prograde_bh_spin_angles = bh_initial_spin_angles[prograde_orb_ang_mom_indices]
#    print(prograde_bh_spin_angles)
    print("Prograde BH initial generations")
    prograde_bh_generations = bh_initial_generations[prograde_orb_ang_mom_indices]

    #4 Test Binary formation
    #Number of binary properties that we want to record (e.g. R1,R2,M1,M2,a1,a2,theta1,theta2,sep,com,t_gw,merger_flag,time of merger, gen_1,gen_2, bin_ang_mom)o
    number_of_bin_properties = len(binary_field_names.split())+1
    integer_nbinprop = int(number_of_bin_properties)
    bin_index = 0
    int_bin_index=int(bin_index)
    test_bin_number = n_bins_max
    integer_test_bin_number = int(test_bin_number)
    number_of_mergers = 0
    #int_num_mergers = int(number_of_mergers)

    #Set up empty initial Binary array
    #Initially all zeros, then add binaries plus details as appropriate
    binary_bh_array = np.zeros((integer_nbinprop,integer_test_bin_number))
    #Set up normalization for t_gw
    norm_t_gw = tgw.normalize_tgw(mass_smbh)
    print("Scale of t_gw (yrs)=", norm_t_gw)
    
    # Set up merger array (identical to binary array)
    #number_of_merger_properties = 16.0
#    num_of_mergers=4.0
    #int_merg_props=int(number_of_merger_properties)
    #int_n_merg=int(num_of_mergers)
    merger_array = np.zeros((integer_nbinprop,integer_test_bin_number))
    
    #Set up output array (mergerfile)
    nprop_mergers=len(mergerfile.names_rec)
    integer_nprop_merge=int(nprop_mergers)
    merged_bh_array = np.zeros((integer_nprop_merge,integer_test_bin_number))
    merged_bh_rec_array = np.empty((integer_nbinprop, integer_test_bin_number), dtype=mergerfile.dtype_rec)
    #Start Loop of Timesteps
    print("Start Loop!")
    time_passed = initial_time
    print("Initial Time(yrs) = ",time_passed)

    n_mergers_so_far = 0
    n_timestep_index = 0
    while time_passed < final_time:
        # Record 
        if not(opts.no_snapshots):
            n_bh_out_size = len(prograde_bh_locations)
            svals = list(map( lambda x: x.shape,[prograde_bh_locations, prograde_bh_masses, prograde_bh_spins, prograde_bh_spin_angles, prograde_bh_generations[:n_bh_out_size]]))
            # Single output:  does work
            np.savetxt("output_bh_single_{}.dat".format(n_timestep_index), np.c_[prograde_bh_locations.T, prograde_bh_masses.T, prograde_bh_spins.T, prograde_bh_spin_angles.T,prograde_bh_generations[:n_bh_out_size].T], header="r_bh m a theta gen")
            # Binary output: does not work
            np.savetxt("output_bh_binary_{}.dat".format(n_timestep_index),binary_bh_array[:,:n_mergers_so_far+1].T,header=binary_field_names)
            n_timestep_index +=1

        
        #Migrate
        # First if feedback present, find ratio of feedback heating torque to migration torque
        #print("feedback",feedback)
        if feedback > 0:
            ratio_heat_mig_torques = feedback_hankla21.feedback_hankla(prograde_bh_locations, surf_dens_func, frac_Eddington_ratio, alpha)
        else:
            ratio_heat_mig_torques = np.ones(len(prograde_bh_locations))   

        prograde_bh_locations = type1.type1_migration(mass_smbh , prograde_bh_locations, prograde_bh_masses, disk_surface_density, disk_aspect_ratio, timestep, ratio_heat_mig_torques, trap_radius)
        #Accrete
        prograde_bh_masses = changebhmass.change_mass(prograde_bh_masses, frac_Eddington_ratio, mass_growth_Edd_rate, timestep)
        #Spin up    
        prograde_bh_spins = changebh.change_spin_magnitudes(prograde_bh_spins, frac_Eddington_ratio, spin_torque_condition, timestep)
        #Torque spin angle
        prograde_bh_spin_angles = changebh.change_spin_angles(prograde_bh_spin_angles, frac_Eddington_ratio, spin_torque_condition, spin_minimum_resolution, timestep)
        #Calculate size of Hill sphere
        bh_hill_sphere = hillsphere.calculate_hill_sphere(prograde_bh_locations, prograde_bh_masses, mass_smbh)
        #Test for encounters within Hill sphere
        print("Time passed", time_passed)
        print("Number of binaries=", bin_index)
        #print("Initial binary array", binary_bh_array)
        #If binary exists, harden it. Add a thing here.
        if bin_index > 0:
            #Evolve binaries. 
            #Migrate binaries
            binary_bh_array = evolve.com_migration(binary_bh_array, disk_surface_density, disk_aspect_ratio, timestep, integer_nbinprop, bin_index)
            #Accrete gas onto binaries
            binary_bh_array = evolve.change_bin_mass(binary_bh_array, frac_Eddington_ratio, mass_growth_Edd_rate, timestep, integer_nbinprop, bin_index)
            #Spin up binary components
            binary_bh_array = evolve.change_bin_spin_magnitudes(binary_bh_array, frac_Eddington_ratio, spin_torque_condition, timestep, integer_nbinprop, bin_index)
            #Torque binary spin components
            binary_bh_array = evolve.change_bin_spin_angles(binary_bh_array, frac_Eddington_ratio, spin_torque_condition, spin_minimum_resolution, timestep, integer_nbinprop, bin_index)

            #Check and see if merger flagged (row 11, if negative)
            merger_flags=binary_bh_array[11,:]
            any_merger=np.count_nonzero(merger_flags) 
            if verbose:
                print(merger_flags)
            merger_indices = np.where(merger_flags < 0.0)
            if isinstance(merger_indices,tuple):
                merger_indices = merger_indices[0]
            if verbose:
                print(merger_indices)
            #print(binary_bh_array[:,merger_indices])
            if any_merger > 0:
                print("Merger!")
                #Calculate merger properties
                mass_1 = binary_bh_array[2,merger_indices]
                mass_2 = binary_bh_array[3,merger_indices]
                spin_1 = binary_bh_array[4,merger_indices]
                spin_2 = binary_bh_array[5,merger_indices]
                angle_1 = binary_bh_array[6,merger_indices]
                angle_2 = binary_bh_array[7,merger_indices]
                bin_ang_mom = binary_bh_array[16,merger_indices]

                merged_mass = tichy08.merged_mass(mass_1, mass_2, spin_1, spin_2)
                merged_spin = tichy08.merged_spin(mass_1, mass_2, spin_1, spin_2, bin_ang_mom)
                merged_chi_eff = chieff.chi_effective(mass_1, mass_2, spin_1, spin_2, angle_1, angle_2, bin_ang_mom)
#                merged_bh_rec_array = mergerfile.extend_rec_merged_bh(merged_bh_rec_array, n_mergers_so_far,  merger_indices,merged_chi_eff,merged_mass,merged_spin,nprop_mergers,number_of_mergers)
                merged_bh_array = mergerfile.merged_bh(merged_bh_array,binary_bh_array, merger_indices,merged_chi_eff,merged_mass,merged_spin,nprop_mergers,number_of_mergers)
                
                

                merger_array[:,merger_indices] = binary_bh_array[:,merger_indices]
                #print(merger_array)
                #Reset merger marker to zero
                #n_mergers_so_far=int(number_of_mergers)
                #Remove merged binary from binary array. Delete column where merger_indices is the label.
                print("!Merger properties!",binary_bh_array[:,merger_indices],merger_array[:,merger_indices],merged_bh_array)
                binary_bh_array=np.delete(binary_bh_array,merger_indices,1)
                
                #binary_bh_array[:,merger_indices] = 0.0
                #binary_bh_array[11,n_mergers_so_far] = 0
                
                #Reduce number of binaries by number of mergers
                bin_index = bin_index - len(merger_indices)
                print("bin index",bin_index)
                #Find relevant properties of merged BH to add to single BH arrays
                num_mergers_this_timestep = len(merger_indices)
                
                print("num mergers this timestep",num_mergers_this_timestep)
                print("n_mergers_so_far",n_mergers_so_far)    
                for i in range (0,num_mergers_this_timestep):
                    merged_bh_com = merged_bh_array[0,n_mergers_so_far + i]
                    merged_mass = merged_bh_array[1,n_mergers_so_far + i]
                    merged_spin = merged_bh_array[3,n_mergers_so_far + i]
                    merged_spin_angle = merged_bh_array[4,n_mergers_so_far + i]
                #New bh generation is max of generations involved in merger plus 1
                    merged_bh_gen = np.maximum(merged_bh_array[11,n_mergers_so_far + i],merged_bh_array[12,n_mergers_so_far + i]) + 1.0 
                print("Merger at=",merged_bh_com,merged_mass,merged_spin,merged_spin_angle,merged_bh_gen)
                # Add to number of mergers
                n_mergers_so_far += len(merger_indices)
                number_of_mergers += len(merger_indices)

                # Append new merged BH to arrays of single BH locations, masses, spins, spin angles & gens
                prograde_bh_locations = np.append(prograde_bh_locations,merged_bh_com)
                prograde_bh_masses = np.append(prograde_bh_masses,merged_mass)
                prograde_bh_spins = np.append(prograde_bh_spins,merged_spin)
                prograde_bh_spin_angles = np.append(prograde_bh_spin_angles,merged_spin_angle)
                prograde_bh_generations = np.append(prograde_bh_generations,merged_bh_gen)
                sorted_prograde_bh_locations=np.sort(prograde_bh_locations)
                if verbose:
                    print("New BH locations", sorted_prograde_bh_locations)
                print("Merger Flag!")
                print(number_of_mergers)
                print("Time ", time_passed)
                if verbose:
                    print(merger_array)
            else:                
                # No merger
                # Harden binary
                binary_bh_array = baruteau11.bin_harden_baruteau(binary_bh_array,integer_nbinprop,mass_smbh,timestep,norm_t_gw,bin_index,time_passed)
                print("Harden binary")
                print("Time passed = ", time_passed)
                if bin_index>0: # verbose:
                    print(" BH binaries ", bin_index,  binary_bh_array[:,:int(bin_index)].shape)
                    print(binary_bh_array[:,:int(bin_index)].T)  # this makes printing work as expected
        else:
            
            # No Binaries present in bin_array. Nothing to do.
        


        #If a close encounter within mutual Hill sphere add a new Binary

            close_encounters = hillsphere.encounter_test(prograde_bh_locations, bh_hill_sphere)
            print(close_encounters)
            if len(close_encounters) > 0:
                print("Make binary at time ", time_passed)
                sorted_prograde_bh_locations = np.sort(prograde_bh_locations)
                sorted_prograde_bh_location_indices = np.argsort(prograde_bh_locations)
                number_of_new_bins = (len(close_encounters))/2            
                binary_bh_array = add_new_binary.add_to_binary_array(binary_bh_array, prograde_bh_locations, prograde_bh_masses, prograde_bh_spins, prograde_bh_spin_angles, prograde_bh_generations, close_encounters, bin_index, retro)
                bin_index = bin_index + number_of_new_bins
                bh_masses_by_sorted_location = prograde_bh_masses[sorted_prograde_bh_location_indices]
                bh_spins_by_sorted_location = prograde_bh_spins[sorted_prograde_bh_location_indices]
                bh_spin_angles_by_sorted_location = prograde_bh_spin_angles[sorted_prograde_bh_location_indices]
                #Delete binary info from individual BH arrays
                sorted_prograde_bh_locations = np.delete(sorted_prograde_bh_locations, close_encounters)
                bh_masses_by_sorted_location = np.delete(bh_masses_by_sorted_location, close_encounters)
                bh_spins_by_sorted_location = np.delete(bh_spins_by_sorted_location, close_encounters)
                bh_spin_angles_by_sorted_location = np.delete(bh_spin_angles_by_sorted_location, close_encounters)
                #Reset arrays
                prograde_bh_locations = sorted_prograde_bh_locations
                prograde_bh_masses = bh_masses_by_sorted_location
                prograde_bh_spins = bh_spins_by_sorted_location
                prograde_bh_spin_angles = bh_spin_angles_by_sorted_location

        #Iterate the time step
        #Empty close encounters
        empty = []
        close_encounters = np.array(empty)

        #After this time period, was there a disk capture via orbital grind-down?
        capture = time_passed % capture_time
        if capture == 0:
            bh_capture_location = setupdiskblackholes.setup_disk_blackholes_location(1, outer_capture_radius)
            bh_capture_mass = setupdiskblackholes.setup_disk_blackholes_masses(1, mode_mbh_init, max_initial_bh_mass, mbh_powerlaw_index)
            bh_capture_spin = setupdiskblackholes.setup_disk_blackholes_spins(1, mu_spin_distribution, sigma_spin_distribution)
            bh_capture_spin_angle = setupdiskblackholes.setup_disk_blackholes_spin_angles(1, bh_capture_spin)
            print("CAPTURED BH",bh_capture_location,bh_capture_mass,bh_capture_spin,bh_capture_spin_angle)
            # Append captured BH to existing singleton arrays. Assume prograde and 1st gen BH.
            prograde_bh_locations = np.append(prograde_bh_locations,bh_capture_location) 
            prograde_bh_masses = np.append(prograde_bh_masses,bh_capture_mass)
            prograde_bh_spins = np.append(prograde_bh_spins,bh_capture_spin)
            prograde_bh_spin_angles = np.append(prograde_bh_spin_angles,bh_capture_spin_angle) 
            prograde_bh_generations = np.append(prograde_bh_generations,1)



        time_passed = time_passed + timestep
    #End Loop of Timesteps at Final Time, end all changes & print out results
    
    print("End Loop!")
    print("Final Time (yrs) = ",time_passed)
    if verbose:
        print("BH locations at Final Time")
        print(prograde_bh_locations)
    print("Number of binaries = ",bin_index)
    print("Total number of mergers = ",number_of_mergers)
    print("Mergers", merged_bh_array.shape)
    if True and number_of_mergers > 0: #verbose:
        print(merged_bh_array[:,:number_of_mergers].T)
        
    np.savetxt(opts.fname_output_mergers, merged_bh_array[:,:number_of_mergers].T, header=merger_field_names)


if __name__ == "__main__":
    main()
