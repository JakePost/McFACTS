import numpy as np

def setup_disk_blackholes_location(n_bh,disk_outer_radius):
    #Return an array of BH locations distributed randomly uniformly in disk
    rng=np.random.random
    integer_nbh=int(n_bh)
    bh_initial_locations=disk_outer_radius*rng(integer_nbh)
    return bh_initial_locations

def setup_disk_blackholes_masses(n_bh,mode_mbh_init,max_initial_bh_mass,mbh_powerlaw_index):
    #Return an array of BH initial masses for a given powerlaw index and max mass
    integer_nbh=int(n_bh)
    bh_initial_masses=(np.random.pareto(mbh_powerlaw_index,integer_nbh)+1)*mode_mbh_init
    #impose maximum mass condition
    bh_initial_masses[bh_initial_masses>max_initial_bh_mass]=max_initial_bh_mass
    return bh_initial_masses

def setup_disk_blackholes_spins(n_bh,mu_spin_distribution,sigma_spin_distribution):
    #Return an array of BH initial spin magnitudes for a given mode and sigma of a distribution
    integer_nbh=int(n_bh)
    bh_initial_spins=np.random.normal(mu_spin_distribution,sigma_spin_distribution,integer_nbh)
    return bh_initial_spins

def setup_disk_blackholes_spin_angles(n_bh,bh_initial_spins):
    #Return an array of BH initial spin angles (in radians).
    #Positive (negative) spin magnitudes have spin angles [0,1.57]([1.5701,3.14])rads
    #All BH spin angles drawn from [0,1.57]rads and +1.57rads to negative spin indices
    integer_nbh=int(n_bh)
    bh_initial_spin_indices=np.array(bh_initial_spins)
    negative_spin_indices=np.where(bh_initial_spin_indices < 0.)
    bh_initial_spin_angles=np.random.uniform(0.,1.57,integer_nbh)
    bh_initial_spin_angles[negative_spin_indices]=bh_initial_spin_angles[negative_spin_indices]+1.57
    return bh_initial_spin_angles

def setup_disk_blackholes_orb_ang_mom(n_bh):
    #Return an array of BH initial orbital angular momentum.
    #Assume either fully prograde (+1) or retrograde (-1)
    integer_nbh=int(n_bh)
    random_uniform_number=np.random.random_sample((integer_nbh,))
    bh_initial_orb_ang_mom=(2.0*np.around(random_uniform_number))-1.0
    return bh_initial_orb_ang_mom

