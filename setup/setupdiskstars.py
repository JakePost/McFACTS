import numpy as np


def setup_disk_stars_location(rng, n_stars, disk_outer_radius):
    #Return an array of star locations distributed randomly uniformly in disk
    integer_nstars = int(n_stars)
    stars_initial_locations = disk_outer_radius*rng.random(integer_nstars)
    return stars_initial_locations


def setup_disk_stars_masses(rng, n_stars,mode_mstar_init,max_initial_star_mass,mstar_powerlaw_index):
    #Return an array of star initial masses for a given powerlaw index and max mass
    integer_nstars = int(n_stars)
    stars_initial_masses = (rng.pareto(mstar_powerlaw_index,integer_nstars)+1)*mode_mstar_init
    #impose maximum mass condition
    stars_initial_masses[stars_initial_masses > max_initial_star_mass] = max_initial_star_mass
    return stars_initial_masses


def setup_disk_stars_spins(rng, n_stars, mu_spin_distribution, sigma_spin_distribution):
    #Return an array of star initial spin magnitudes for a given mode and sigma of a distribution
    integer_nstars = int(n_stars)
    stars_initial_spins = rng.normal(mu_spin_distribution, sigma_spin_distribution, integer_nstars)
    return stars_initial_spins


def setup_disk_stars_spin_angles(rng, n_stars, stars_initial_spins):
    #Return an array of star initial spin angles (in radians).
    #Positive (negative) spin magnitudes have spin angles [0,1.57]([1.5701,3.14])rads
    #All star spin angles drawn from [0,1.57]rads and +1.57rads to negative spin indices
    integer_nstars = int(n_stars)
    stars_initial_spin_indices = np.array(stars_initial_spins)
    negative_spin_indices = np.where(stars_initial_spin_indices < 0.)
    stars_initial_spin_angles = rng.uniform(0.,1.57,integer_nstars)
    stars_initial_spin_angles[negative_spin_indices] = stars_initial_spin_angles[negative_spin_indices] + 1.57
    return stars_initial_spin_angles


def setup_disk_stars_orb_ang_mom(rng, n_stars):
    #Return an array of star initial orbital angular momentum.
    #Assume either fully prograde (+1) or retrograde (-1)
    integer_nstars = int(n_stars)
    random_uniform_number = rng.random((integer_nstars,))
    stars_initial_orb_ang_mom = (2.0*np.around(random_uniform_number)) - 1.0
    return stars_initial_orb_ang_mom

def setup_disk_stars_eccentricity_thermal(rng, n_stars):
    # Return an array of star orbital eccentricities
    # For a thermal initial distribution of eccentricities, select from a uniform distribution in e^2.
    # Thus (e=0.7)^2 is 0.49 (half the eccentricities are <0.7). 
    # And (e=0.9)^2=0.81 (about 1/5th eccentricities are >0.9)
    # So rnd= draw from a uniform [0,1] distribution, allows ecc=sqrt(rnd) for thermal distribution.
    # Thermal distribution in limit of equipartition of energy after multiple dynamical encounters
    integer_nstars = int(n_stars)
    random_uniform_number = rng.random((integer_nstars,))
    stars_initial_orb_ecc = np.sqrt(random_uniform_number)
    return stars_initial_orb_ecc

def setup_disk_stars_eccentricity_uniform(rng, n_stars):
    # Return an array of star orbital eccentricities
    # For a uniform initial distribution of eccentricities, select from a uniform distribution in e.
    # Thus half the eccentricities are <0.5
    # And about 1/10th eccentricities are >0.9
    # So rnd = draw from a uniform [0,1] distribution, allows ecc = rnd for uniform distribution
    # Most real clusters/binaries lie between thermal & uniform (e.g. Geller et al. 2019, ApJ, 872, 165)
    integer_nstars = int(n_stars)
    random_uniform_number = rng.random((integer_nstars,))
    stars_initial_orb_ecc = random_uniform_number
    return stars_initial_orb_ecc

def setup_disk_stars_inclination(rng, n_stars):
    # Return an array of star orbital inclinations
    # Return an initial distribution of inclination angles that are 0.0
    #
    # To do: initialize inclinations so random draw with i <h (so will need to input star_locations and disk_aspect_ratio)
    # and then damp inclination.
    # To do: calculate v_kick for each merger and then the (i,e) orbital elements for the newly merged BH. 
    # Then damp (i,e) as appropriate
    integer_nstars = int(n_stars)
    # For now, inclinations are zeros
    stars_initial_orb_incl = np.zeros((integer_nstars,),dtype = float)
    return stars_initial_orb_incl

def setup_disk_stars_circularized(rng, n_stars,crit_ecc):
    # Return an array of star orbital inclinations
    # Return an initial distribution of inclination angles that are 0.0
    #
    # To do: initialize inclinations so random draw with i <h (so will need to input star_locations and disk_aspect_ratio)
    # and then damp inclination.
    # To do: calculate v_kick for each merger and then the (i,e) orbital elements for the newly merged BH. 
    # Then damp (i,e) as appropriate
    integer_nstars = int(n_stars)
    # For now, inclinations are zeros
    stars_initial_orb_ecc = crit_ecc*np.ones((integer_nstars,),dtype = float)
    return stars_initial_orb_ecc

def setup_disk_nbh(M_nsc,nbh_nstar_ratio,mbh_mstar_ratio,r_nsc_out,nsc_index_outer,mass_smbh,disk_outer_radius,h_disk_average,r_nsc_crit,nsc_index_inner):
    # Return the integer number of BH in the AGN disk as calculated from NSC inputs assuming isotropic distribution of NSC orbits
    # To do: Calculate when R_disk_outer is not equal to the r_nsc_crit
    # To do: Calculate when disky NSC population of BH in plane/out of plane.
    # Housekeeping:
    # Convert outer disk radius in r_g to units of pc. 1r_g =1AU (M_smbh/10^8Msun) and 1pc =2e5AU =2e5 r_g(M/10^8Msun)^-1
    pc_dist = 2.e5*((mass_smbh/1.e8)**(-1.0))
    critical_disk_radius_pc = disk_outer_radius/pc_dist
    #Total average mass of BH in NSC
    M_bh_nsc = M_nsc * nbh_nstar_ratio * mbh_mstar_ratio
    #print("M_bh_nsc",M_bh_nsc)
    #Total number of BH in NSC
    N_bh_nsc = M_bh_nsc / mbh_mstar_ratio
    #print("N_bh_nsc",N_bh_nsc)
    #Relative volumes:
    #   of central 1 pc^3 to size of NSC
    relative_volumes_at1pc = (1.0/r_nsc_out)**(3.0)
    #   of r_nsc_crit^3 to size of NSC
    relative_volumes_at_r_nsc_crit = (r_nsc_crit/r_nsc_out)**(3.0)
    #print(relative_volumes_at1pc)
    #Total number of BH 
    #   at R<1pc (should be about 10^4 for Milky Way parameters; 3x10^7Msun, 5pc, r^-5/2 in outskirts)
    N_bh_nsc_pc = N_bh_nsc * relative_volumes_at1pc * (1.0/r_nsc_out)**(-nsc_index_outer)
    #   at r_nsc_crit
    N_bh_nsc_crit = N_bh_nsc * relative_volumes_at_r_nsc_crit * (r_nsc_crit/r_nsc_out)**(-nsc_index_outer)
    #print("Normalized N_bh at 1pc",N_bh_nsc_pc)
    
    #Calculate Total number of BH in volume R < disk_outer_radius, assuming disk_outer_radius<=1pc.
    
    if critical_disk_radius_pc >= r_nsc_crit:
        relative_volumes_at_disk_outer_radius = (critical_disk_radius_pc/1.0)**(3.0)
        Nbh_disk_volume = N_bh_nsc_pc * relative_volumes_at_disk_outer_radius * ((critical_disk_radius_pc/1.0)**(-nsc_index_outer))          
    else:
        relative_volumes_at_disk_outer_radius = (critical_disk_radius_pc/r_nsc_crit)**(3.0)
        Nbh_disk_volume = N_bh_nsc_crit * relative_volumes_at_disk_outer_radius * ((critical_disk_radius_pc/r_nsc_crit)**(-nsc_index_inner))
     
    # Total number of BH in disk
    Nbh_disk_total = np.rint(Nbh_disk_volume * h_disk_average)
    #print("Nbh_disk_total",Nbh_disk_total)  
    return np.int64(Nbh_disk_total)

