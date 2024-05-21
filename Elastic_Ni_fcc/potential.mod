# NOTE: This script can be modified for different pair styles 
# See in.elastic for more info.

# Choose potential
#pair_style	sw
#pair_coeff * * Si.sw Si

pair_style      zero 7.0
pair_coeff	* *  

fix             integ all nve
python          post_force_callback file animpiuq.py
fix             2 all python/torch 1 post_force post_force_callback
fix_modify      2 energy yes
run             0
# Setup neighbor style
#neighbor 1.0 nsq
#neigh_modify once no every 1 delay 0 check yes
neighbor        1.0 bin
neigh_modify    every 1 delay 0 check yes

# Setup minimization style
min_style	     cg
min_modify	     dmax ${dmax} line quadratic

# Setup output
thermo		1
thermo_style custom step temp pe press pxx pyy pzz pxy pxz pyz lx ly lz vol
thermo_modify norm no
#run             1000
