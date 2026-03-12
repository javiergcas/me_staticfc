ml afni

# Compute Mean signal levels per voxel in each echo
3dTstat -overwrite -prefix jrm.e01.mean.nii.gz pb00.sub-01.r01.e01.tcat+orig
3dTstat -overwrite -prefix jrm.e02.mean.nii.gz pb00.sub-01.r01.e02.tcat+orig
3dTstat -overwrite -prefix jrm.e03.mean.nii.gz pb00.sub-01.r01.e03.tcat+orig

# We will represent each voxel as 3 numbers, which are the mean values.
# Vectors with a very small norm (e.g., 1) can be considered free of signal
3dcalc -overwrite -a jrm.e01.mean.nii.gz -b jrm.e02.mean.nii.gz -c jrm.e03.mean.nii.gz -expr 'sqrt((a-b)^2 + (a-c)^2 + (b-c)^2)' -prefix jrm.euc_dist_emeans.nii.gz
3dcalc -a jrm.euc_dist_emeans.nii.gz -expr 'isnegative(a-1)' -prefix jrm.mask.nii.gz

# Using the mask described as above, we compute the spatial stdev per volume per echo 
3dROIstats -sigma -nomeanout -quiet -mask jrm.mask.nii.gz pb00.sub-01.r01.e01.tcat+orig. | awk {'print $1}' > jrm.thermal.e01.1D
3dROIstats -sigma -nomeanout -quiet -mask jrm.mask.nii.gz pb00.sub-01.r01.e02.tcat+orig. | awk {'print $1}' > jrm.thermal.e02.1D
3dROIstats -sigma -nomeanout -quiet -mask jrm.mask.nii.gz pb00.sub-01.r01.e03.tcat+orig. | awk {'print $1}' > jrm.thermal.e03.1D

# We paste it together to visualize it
paste jrm.thermal.e01.1D jrm.thermal.e02.1D jrm.thermal.e03.1D > jrm.thermal.all.1D
1dplot -one jrm.thermal.all.1D 

