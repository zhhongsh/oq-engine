Classical PSHA QA test with sites_csv
=====================================

============== ===================
checksum32     1,067,610,621      
date           2019-10-23T16:26:42
engine_version 3.8.0-git2e0d8e6795
============== ===================

num_sites = 10, num_levels = 13, num_rlzs = 1

Parameters
----------
=============================== ==================
calculation_mode                'preclassical'    
number_of_logic_tree_samples    1                 
maximum_distance                {'default': 200.0}
investigation_time              50.0              
ses_per_logic_tree_path         1                 
truncation_level                3.0               
rupture_mesh_spacing            2.0               
complex_fault_mesh_spacing      2.0               
width_of_mfd_bin                0.1               
area_source_discretization      10.0              
ground_motion_correlation_model None              
minimum_intensity               {}                
random_seed                     23                
master_seed                     0                 
ses_seed                        42                
=============================== ==================

Input files
-----------
========================= ============================================================
Name                      File                                                        
========================= ============================================================
gsim_logic_tree           `gmpe_logic_tree.xml <gmpe_logic_tree.xml>`_                
job_ini                   `job.ini <job.ini>`_                                        
reqv:active shallow crust `lookup_asc.hdf5 <lookup_asc.hdf5>`_                        
sites                     `qa_sites.csv <qa_sites.csv>`_                              
source_model_logic_tree   `source_model_logic_tree.xml <source_model_logic_tree.xml>`_
========================= ============================================================

Composite source model
----------------------
============ ======= =============== ================
smlt_path    weight  gsim_logic_tree num_realizations
============ ======= =============== ================
simple_fault 1.00000 simple(2)       2               
============ ======= =============== ================

Required parameters per tectonic region type
--------------------------------------------
====== ================================================= =========== ============================= =======================
grp_id gsims                                             distances   siteparams                    ruptparams             
====== ================================================= =========== ============================= =======================
0      '[AbrahamsonSilva2008]' '[CampbellBozorgnia2008]' rjb rrup rx vs30 vs30measured z1pt0 z2pt5 dip mag rake width ztor
====== ================================================= =========== ============================= =======================

Realizations per (GRP, GSIM)
----------------------------

::

  <RlzsAssoc(size=1, rlzs=1)>

Number of ruptures per source group
-----------------------------------
====== ========= ============ ============
grp_id num_sites num_ruptures eff_ruptures
====== ========= ============ ============
0      0.02237   447          447         
====== ========= ============ ============

Slowest sources
---------------
========= ====== ==== ============ ========= ========= ============
source_id grp_id code num_ruptures calc_time num_sites eff_ruptures
========= ====== ==== ============ ========= ========= ============
3         0      S    447          0.00306   0.02237   447         
========= ====== ==== ============ ========= ========= ============

Computation times by source typology
------------------------------------
==== =========
code calc_time
==== =========
S    0.00306  
==== =========

Information about the tasks
---------------------------
================== ======= ====== ======= ======= =======
operation-duration mean    stddev min     max     outputs
SourceReader       0.00495 NaN    0.00495 0.00495 1      
preclassical       0.00332 NaN    0.00332 0.00332 1      
================== ======= ====== ======= ======= =======

Data transfer
-------------
============ ============================================= ========
task         sent                                          received
preclassical params=110.62 KB srcs=1.11 KB srcfilter=223 B 342 B   
============ ============================================= ========

Slowest operations
------------------
====================== ========= ========= ======
calc_44543             time_sec  memory_mb counts
====================== ========= ========= ======
composite source model 0.02991   0.0       1     
total SourceReader     0.00495   0.0       1     
total preclassical     0.00332   0.0       1     
store source_info      0.00220   0.0       1     
aggregate curves       2.182E-04 0.0       1     
====================== ========= ========= ======