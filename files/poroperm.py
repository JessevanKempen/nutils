import pandas as pd
from scipy import stats
import numpy as np
import math

#user input
def get_poroperm(name_strata):
    poroperm = pd.read_excel(r'C:\Users\s141797\OneDrive - TU Eindhoven\Scriptie\nlog_poroperm.xlsx') #for an earlier version of Excel use 'xls'
    columns = ['DEPTH', 'POROSITY','HOR_PERMEABILITY', 'VERT_PERMEABILITY', 'STRAT_UNIT_NM']

    df = pd.DataFrame(poroperm, columns = columns)
    df_strata = df.loc[(df['STRAT_UNIT_NM'] == name_strata ), ['POROSITY','HOR_PERMEABILITY', 'VERT_PERMEABILITY']]

    return df_strata['POROSITY'], df_strata['HOR_PERMEABILITY']

def get_pdf(samples, size):
    mu = np.mean(samples)
    print(mu)
    stddv = np.var(samples) ** 0.5
    print(stddv)
    pdf = stats.lognorm(scale=mu, s=stddv).rvs(size=size)

    return pdf

por, kh = get_poroperm('Delft Sandstone Member')
pdf_por = print(get_pdf(por, 5))