"""
Fetal anatomy
-------------

This module contains a set of functions to estimate fetal biometric measurements
from gestational age:
  - biparietal diameter (BPD) [1]
  - occipitofrontal diameter (OFD) [1]
  - brain volume (BV) [2]
  - crown-rump length (CRL) [3]
  - weight [4]

All functions take a gestational age (in weeks), and optionnally a centile
(between 0.0 and 1.0, default 0.5, corresponding to the median, which is also
the average if a normal distribition is assumed), and return a value in mm.

Cubic polynoms have been fitted using numpy.polyfit to data from [1,3,4], while
[2] already provided a regression formula.

References:
-----------

[1] Snijders, R. J. M. and Nicolaides, K. H. (1994), Fetal biometry at 14-40
weeks' gestation. Ultrasound Obstet Gynecol, 4: 34-48. doi:
10.1046/j.1469-0705.1994.04010034.x

[2] Chiung-Hsin Chang, Chen-Hsiang Yu, Fong-Ming Chang, Huei-Chen Ko, Hsi-Yao
Chen, The assessment of normal fetal brain volume by 3-D ultrasound, Ultrasound
in Medicine & Biology, Volume 29, Issue 9, September 2003, Pages 1267-1272, ISSN
0301-5629, http://dx.doi.org/10.1016/S0301-5629(03)00989-X.

[3] Archie, J.G., Collins, J.S., Lebel, R.R.: Quantitative Standards for Fetal
and Neonatal Autopsy. American Journal of Clinical Pathology 126(2), 256-265 (2006)

[4] Cole TJ, Freeman JV, Preece MA. British 1990 growth reference centiles for weight, height,
body mass index and head circumference fitted by maximum penalized likelihood.
Stat.Med. 1998;17:407-29.
See: Early years - UK-WHO growth charts and resources
http://www.rcpch.ac.uk/child-health/research-projects/uk-who-growth-charts/uk-who-growth-chart-resources-0-4-years/uk-who-0

"""

import numpy as np

__all__ = [ "get_OFD",
            "get_BPD",
            "get_CRL",
            "get_BV",
            "get_weight" ]

def get_OFD( ga, centile=50 ):
    """
    Occipitofrontal diameter (OFD) [1].
    Available centiles: 5, 50, 95.
    """
    polynoms = { 5 : [ -4.37153218e-03,
                        2.76627163e-01,
                        -1.89660847e+00,
                        1.96476341e+01 ],
                 50 : [ -4.97315445e-03,
                         3.19846853e-01,
                         -2.60839214e+00,
                         2.62679565e+01 ],
                 95 : [ -4.96589312e-03,
                         3.11699370e-01,
                         -1.92628231e+00,
                         2.13966173e+01 ] }

    p = np.poly1d( polynoms[centile] )
    return p(ga)
    
def get_BPD( ga, centile=50 ):
    """
    Biparietal diameter (BPD), data from [1].
    Available centiles: 5, 50, 95.
    """
    polynoms = { 5 : [ -3.21975764e-03,
                        2.17264994e-01,
                        -1.81364324e+00,
                        2.00444225e+01 ],
                 50 : [ -3.20651640e-03,
                         2.14380813e-01,
                         -1.47907551e+00,
                         1.87142471e+01 ],
                 95 : [ -3.48009756e-03,
                         2.31948586e-01,
                         -1.57743704e+00,
                         2.01949509e+01 ] }

    p = np.poly1d( polynoms[centile] )
    return p(ga)

def get_CRL( ga, centile=50 ):
    """
    Crown-rump length (CRL), data from [3].
    Available centiles: 2, 50, 98.
    """
    polynoms = { 2 : [  1.20433358e-02,
                        -1.02689224e+00,
                        3.51311733e+01,
                        -2.42715863e+02 ],
                 50 : [  2.44294544e-03,
                         -2.91836566e-01,
                         1.97213860e+01,
                         -1.26958302e+02 ],
                 98 : [ -7.15744487e-03,
                         4.43219104e-01,
                         4.31159874e+00,
                         -1.12007417e+01 ] }
    
    p = np.poly1d( polynoms[centile] )
    return p(ga)

def get_BV( ga ):
    """
    Brain volume (BV), regression formula from [2].
    Only median value (centile=0.5) available.
    """
    # mL to mm3 , 1 ml is 1 cm^3
    return (-171.48036 + 4.8079*ga + 0.29521*ga**2)*1000

def get_weight( ga, centile=50, sex="both" ):
    """
    Weight, data from [4].
    Available centiles: 2, 50, 98.
    Available gender: 'boys', 'girls', 'both'.
    """
    polynoms = { "boys" : { 2 : [ -2.08868860e-04,
                                   2.55956262e-02,
                                   -8.49527157e-01,
                                   9.00704935e+00 ],
                            50 : [ -4.19159491e-04,
                                    4.46286621e-02,
                                    -1.36487959e+00,
                                    1.35597364e+01 ],
                            98 : [ -6.55372646e-04,
                                    6.63045669e-02,
                                    -1.96390035e+00,
                                    1.89458509e+01 ] },
                 "girls" : { 2 : [ -2.15228225e-04,
                                    2.64071683e-02,
                                    -8.81181502e-01,
                                    9.32074109e+00 ],
                             50 : [ -4.03205565e-04,
                                     4.28501366e-02,
                                     -1.30778086e+00,
                                     1.29389572e+01 ],
                             98 : [ -6.60737642e-04,
                                     6.61817107e-02,
                                     -1.94827924e+00,
                                     1.86579947e+01 ] },
                 "both" : { 2 : [ -2.12048543e-04,
                                   2.60013973e-02,
                                   -8.65354329e-01,
                                   9.16389522e+00 ],
                            50 : [ -4.11182528e-04,
                                    4.37393993e-02,
                                    -1.33633022e+00,
                                    1.32493468e+01 ],
                            98 : [ -6.58055144e-04,
                                    6.62431388e-02,
                                    -1.95608979e+00,
                                    1.88019228e+01] }
                 }

    p = np.poly1d( polynoms[sex][centile] )
    return p(ga)
                   
