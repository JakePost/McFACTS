import numpy as np


def calculate_hill_sphere(prograde_bh_locations, prograde_bh_masses, mass_smbh):
    #Return the Hill sphere radius (R_Hill) for an array of prograde BH where
    # R_Hill=a(q/3)^1/3 where a=semi-major axis, q=m_bh/M_SMBH
    
    bh_smbh_mass_ratio = prograde_bh_masses/(3.0*mass_smbh)
    mass_ratio_factor = (bh_smbh_mass_ratio)**(1./3.)
    bh_hill_sphere = prograde_bh_locations*mass_ratio_factor
    #Return the BH Hill sphere radii for all orbiters. Prograde should have much larger Hill sphere
    return bh_hill_sphere

def encounter_test(prograde_bh_locations, bh_hill_sphere):
    #Using Hill sphere size and BH locations see if there are encounters within the Hill sphere
    # return indices of BH involved.

    # First sort the prograde bh locations in order from inner disk to outer disk
    sorted_bh_locations = np.sort(prograde_bh_locations)
    #Returns the indices of the original array in order, to get the sorted array
    sorted_bh_location_indices = np.argsort(prograde_bh_locations)
   
    #Find the appropriate (sorted) Hill sphere radii
    sorted_hill_spheres = bh_hill_sphere[sorted_bh_location_indices]

    # Find the distances between [r1,r2,r3,r4,..] as [r2-r1,r3-r2,r4-r3,..]=[delta1,delta2,delta3..]
    separations = np.diff(sorted_bh_locations)

    # Note that separations are -1 of length of bh_locations
    # Take 1st location off locations array
    sorted_bh_locations_minus_first_element = sorted_bh_locations[1:len(sorted_bh_locations)]
    #Take last location off locations array
    sorted_bh_locations_minus_last_element = sorted_bh_locations[0:len(sorted_bh_locations)-1]

    # Separations are -1 of length of Hill_sphere array
    # Take 1st Hill sphere off Hill sphere array
    sorted_hill_spheres_minus_first = sorted_hill_spheres[1:len(sorted_hill_spheres)]
    # Take last Hill sphere off Hill sphere array
    sorted_hill_spheres_minus_last = sorted_hill_spheres[0:len(sorted_hill_spheres)-1]

    # Compare the Hill sphere distance for each BH with separation to neighbor BH
    #so we compare e.g. r2-r1 vs R_H1, r3-r2 vs R_H2
    comparison_distance_inwards = separations-sorted_hill_spheres_minus_last
    # and e.g. compare r2-r1 vs R_H2, r3-r2 vs R_H3
    comparison_distance_outwards = separations-sorted_hill_spheres_minus_first

    index_in = np.where(comparison_distance_inwards < 0)
    if isinstance(index_in,tuple):
        index_in = index_in[0]
    # E.g say r3-r2 <R_H2 then we'll want the info for the BH at r2 & r3. (i,i+1)
    index_out = np.where(comparison_distance_outwards < 0)
    if isinstance(index_out,tuple):
        index_out = index_out[0]
    #E.g. say r3-r2 <R_H3 then we'll want the info for the BH at r3 and r2 (i,i-1)
    length_index_in = len(index_in)
    length_index_out = len(index_out)

    new_indx_in = list(range(2*len(index_in)))
    new_indx_out = list(range(2*len(index_out)))
    #new_indx_in_bin_array = np.array[2*len(index_in),2]
    #temp_new_bin_array = np.ndarray[2*length_index_in,2]

    for ind in range(length_index_in):
        temp_index = index_in[ind]
        new_indx_in[2*ind] = temp_index
        new_indx_in[(2*ind)+1] = temp_index+1
    #    temp_new_bin_array = 
    print("new_indx_in",new_indx_in)
    temp_new_bin_in_array=np.reshape(new_indx_in,(len(index_in),2))
    print("ordered as bins",temp_new_bin_in_array)
    
    test_remove_bin = new_indx_in
    
    # Dynamics Here! Potential Double-binary or triple interaction!
    # For now! 
    # 0. Construct array of pairs of indices.
    # 1. Select binaries based on distance
    #     For binaries [i-1,i], [i,i+1] compare distance r(i)-r(i-1) to r(i+1)-r(i) and select smallest.
    #     Remove larger distane pair. E.g. if [i+1,i] is smaller, remove [i-1,i]
    #    But want to calculate 
    # 2. Fractional R_Hill for [i-1,i] vs [i,i+1]. Smaller fractional Hill radius wins
    # 3. TO DO Write a module for DiLaurentii+22 or Rowan+22 or LANL+22 phase space encounter and apply to all encounters 
    #          over timestep (10kyrs; assume random phase & number of encounters during timestep; pick randomly 
    #          from phase plots.) Also look at LANL group papers on binding energy of encounter.
    # 4. Ideally, consider the triple-dynamics encounter 
    
    # Search for repeat indices in binary array 
    unique_element,unique_index,unique_ct = np.unique(temp_new_bin_in_array,return_inverse = True,return_counts = True)
    repeats = unique_ct > 1
    
    repeat_indices = unique_element[repeats]
    
    duplicate_indices,dup_index,dup_ct = np.unique(unique_index, return_inverse = True, return_counts = True)
    rep_idx = dup_ct >1
    dupl_indices = duplicate_indices[rep_idx]
    print("indices of repeats",dupl_indices)
    

    #return all those elements that only occur once
    new_array = unique_element[unique_ct == 1]
    print("new array",new_array)
    #return the index where the element occurs first e.g. if element 59 occurs at index 9,10 in the array and
    #element 60 occurs at index 11,12, then index returned is =[...8,9,11,13,...]
    unique_values,indices_list = np.unique(temp_new_bin_in_array, return_index=True)
    #unique_values,repeat_indices_list = np.unique(temp_new_bin_in_array, )
    print("indices list",indices_list)   

    
    print("repeat_indices",repeat_indices)
    #test_remove_bin=np.delete(test_remove_bin,repeat_indices)
    #print("test remove bin",test_remove_bin)

    
    #Outer bin (i,i+1) here
    temp_dist_outer_bin_in = comparison_distance_inwards[repeat_indices]
    temp_dist_outer_bin_out = comparison_distance_outwards[repeat_indices]
    smallest_sep_outer_bin = np.fmin(temp_dist_outer_bin_in,temp_dist_outer_bin_out)
    print("outer bin",smallest_sep_outer_bin)
    #Inner bin (i-1,i) here
    temp_dist_in_minus1 = comparison_distance_inwards[repeat_indices-1]
    temp_dist_out_minus1 = comparison_distance_outwards[repeat_indices-1]
    temp_dist_inner_bin = np.fmin(temp_dist_in_minus1,temp_dist_out_minus1)
    print("inner bin",temp_dist_inner_bin)
    #print("repeat_indices",repeat_indices)
    #These are the indices of binaries to be deleted. 
    #element_to_be_kept = np.where(smallest_sep_outer_bin < temp_dist_inner_bin, repeat_indices, repeat_indices + 1)
    #element_to_be_deleted = np.where(smallest_sep_outer_bin > temp_dist_inner_bin,repeat_indices,repeat_indices + 1)

    new_test_remove_bin = test_remove_bin
    print("test_remove_bin",new_test_remove_bin)
    #set up array of indices to be removed
    idx_array=np.zeros(len(2*dupl_indices))

    for ind in range(len(dupl_indices)):
        idx = dupl_indices[ind] - (ind)
        value=2*ind
        # If outer bin to be kept    
        if smallest_sep_outer_bin[ind] < temp_dist_inner_bin[ind]:
            #delete inner bin 
            remove_indxs=[idx - 1,idx]
            
            #idx_array[value]=idx - 1
            #idx_array[value+1]=idx
            new_test_remove_bin = np.delete(new_test_remove_bin,remove_indxs)
            print("test remove",new_test_remove_bin)
        else:
            #delete outer bin
            remove_indxs=[idx+1,idx+2]

            #idx_array[value]=idx+1
            #idx_array[value]=idx+2

            new_test_remove_bin = np.delete(new_test_remove_bin,remove_indxs)
            print("test remove",new_test_remove_bin)
    
    print("idx_array",idx_array)

    # Say [10,11,15] Not 9-10;so 10-11 then 13:14.  
    # New=[0:9,11:14,15:end]
    #while ind in range(2*dupl_indices):
    #    new_array=test_remove_bin[1:ind(i),ind(i+2):ind(i+3)-1,ind(i)]
    
    #CHANGE THIS going forward: For removing repeats. Compute mutual Hill sphere (R_H=((M1+M_2)/M_SMBH)^{1/3}) and
    # see how small a fraction of the mutual Hill sphere, the binary separation *would* be.
    #E.g. (r2-R1)=0.5R_H(m1,m2) vs (r3-r2)=0.2R_H(m3,m2) form the tighter binary (r3,r2)

    #Compare nearest neighbour separations for BH in repeat_indices
    # E.g. separations =[r2-r1,r3-r2,r3-4]. If BH at r2 repeats in binary array 
    # e.g. bin_array=[[r1,r2] [r2,r3]..] then compare separations[r2-r1] to separations[r3-r2]
    # If r2-r1 < r3-r2 then make [r1,r2] the binary and remove [r2,r3]
    
    #if separations[repeat_indices] < separations[repeat_indices + 1]:
    #    temp_new_bin_in_array = np.delete(temp_new_bin_in_array,repeat_indices + 1)
    #    print("temp_bin_array",temp_new_bin_in_array)

#    for j in range(2*len(index_in)):    
#        temp_bin_in = new_indx_in_bin_array[temp_index[]]
   

#    for j in range(2*len(index_in)):    
#        temp_bin_in = new_indx_in_bin_array[temp_index[]]
   
    for ind in range(length_index_out):
        temp_index = index_out[ind]
        new_indx_out[2*ind] = temp_index
        new_indx_out[(2*ind)+1] = temp_index+1

    print("new_indx_out",new_indx_out)    
    temp_new_bin_out_array=np.reshape(new_indx_out,(len(index_out),2))
    print("ordered as bins",temp_new_bin_out_array)

    new_indxs = new_indx_in+new_indx_out
    #rindx = np.sort(new_indxs)
    result = np.asarray(new_test_remove_bin)
    #result = np.asarray(new_indx_in)
    sorted_in_result = np.sort(result)

    new_result = np.asarray(new_indx_out)
    sorted_out_result = np.sort(new_result)
    print("sorted in result",sorted_in_result)
    print("sorted out result",sorted_out_result)
    # Concatenate the two lists, and remove duplicates
    final_bin_indices = np.array(list(set(list(sorted_in_result) + list(sorted_out_result))))
    sorted_final_bin_indices = np.sort(final_bin_indices)
    print("total final bin indices",sorted_final_bin_indices)
    #print("check if sorted_in & sorted_out arrays are the same")
    check = np.array_equiv(sorted_in_result, sorted_out_result)
    #print(check)
    # Return the indices of those elements in separation array <0
    # (ie BH are closer than 1 R_Hill)
    # In inwards case, r_i+1 -r_i <R_H_i, so relevant BH indices are i,i+1
   
    # In outwards case, r_i - r_i-1 <R_H_i so relevant BH indices are i,i-1
    final_1d_indx_array = sorted_in_result.flatten()
    sorted_final_1d_indx_array = np.sort(final_1d_indx_array)

    if len(sorted_final_1d_indx_array) > 0:
         print("Binary", sorted_final_1d_indx_array)

    return sorted_final_1d_indx_array
