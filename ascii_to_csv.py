import numpy as np 
from astropy.io import ascii
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table, Column, MaskedColumn
import pandas as pd
import cProfile

def load_data():

    young_data = np.loadtxt('./young.ascii',skiprows=1,delimiter=',',
            dtype={'names': ('DR2ID', 'RA', 'DEC'),
                'formats': ('S30', 'f8', 'f8')})

    old_data = np.loadtxt('./old.ascii',skiprows=1,delimiter=',',
            dtype={'names': ('DR2ID', 'RA', 'DEC'),
                'formats': ('S30', 'f8', 'f8')})

    test_data = np.loadtxt('./Gaia_short.ascii',skiprows=1,delimiter=',',
            dtype={'names': ('DR2Name','RA_ICRS','E_RA_ICRS','DE_ICRS','E_DE_ICRS','source',
                             'plx','E_PLX','PMRA','E_PMRA','PMDE','E_PMDE',
                             'CHI2AL','EPSI','SEPSI','dup','Gmag','e_Gmag',
                             'BPMAG','e_BPMAG','rpmag','e_rpmag','bp_rp','bp_g',
                             'g_rp','rv','e_rv','glon'),
            'formats': ('S30', 'f8', 'f8', 'f8', 'f8', 'f8',
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                        'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                        'f8', 'f8', 'f8', 'f8')})

    headers = ['DR2Name','RA_ICRS','E_RA_ICRS','DE_ICRS','E_DE_ICRS','source',
                                'plx','E_PLX','PMRA','E_PMRA','PMDE','E_PMDE',
                                'CHI2AL','EPSI','SEPSI','dup','Gmag','e_Gmag',
                                'BPMAG','e_BPMAG','rpmag','e_rpmag','bp_rp','bp_g',
                                'g_rp','rv','e_rv','glon']

    my_dict = {header: test_data[header] for header in headers}
    df = pd.DataFrame.from_dict(my_dict)

    
    return old_data, young_data, test_data


def reduce_data_columns(test_data):

    headers = ['DR2Name','RA_ICRS','DE_ICRS',
                    'plx', 'Gmag','BPMAG','rpmag','bp_rp',
                        'bp_g','g_rp','rv','glon']
    
    my_dict = {header: test_data[header] for header in headers}

    df = pd.DataFrame.from_dict(my_dict)

    return df


def make_old_young(totalDf,youngData,oldData):
    ymask = np.isin(totalDf['DR2Name'], youngData['DR2ID'],assume_unique=True)
    omask = np.isin(totalDf['DR2Name'], oldData['DR2ID'],assume_unique=True)
    
    young_df = totalDf[ymask]
    old_df   = totalDf[omask]
    
    ymask_ind = [ind for ind, val in enumerate(ymask) if val]
    omask_ind = [ind for ind, val in enumerate(omask) if val]
    
    mask_full = np.full((len(omask)),'?')
    mask_full[ymask_ind] = 1
    mask_full[omask_ind] = 0

    totalDf['young'] = mask_full

    return young_df, old_df, totalDf


def main():
    old_data, young_data, test_data = load_data()
    total_df = reduce_data_columns(test_data)

    young_df, old_df, total_df= make_old_young(total_df,young_data,old_data)
    total_df.to_csv("Gaia_total_df.csv", sep=',',index=False)
    young_df.to_csv("Gaia_young_df.csv", sep=',',index=False)
    old_df.to_csv("Gaia_old_df.csv", sep=',',index=False)

main()
#cProfile.run('load_data()')


