#Ordering tools
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import collections
# import matplotlib.pyplot as plt

## Parameters txt file reader
def read_from_txt ( filename ):

    # Open the parameter file to read
    with open(filename, 'r') as json_file:
            data = json.load(json_file)

    return data['aquifer'][0], data['well'][0]

## Generates a Parameters txt file
def generate_txt( filename):
    import json

    data = {}
    data['aquifer'] = []
    data['aquifer'].append({
        'd_top': 2387,
        'labda': 0.031,
        'H': 2528-2387,
        'T_surface': 20+273,
        'porosity': 0.2,
        'permeability': 9e-10,
        'rho_f': 996.9,
        'rho_s': 2400,
        'viscosity': 0.0003142,
        'K': 4e-13,
        'cpf' : 4183,
        'cps' : 870,
        'labdas' : 4.2, # thermal conductivity solid [W/mK]
        'labdaf': 0.663, # thermal conductivity fluid [W/mK]
        'saltcontent': 0.155, # [kg/l]
        'pref': 225e5,
        'Tref': 90 + 273,
        'g' : 9.81,
        'rw': 0.1,
        'rmax': 1000,
        'porosity': 0.046,
        'Q': 250 / 3600,
        'L': 1000,
    })
    data['well'] = []
    data['well'].append({
        'r': 0.126, #0.076,
        'Q': 250/3600,
        'L': 1000,
        'Ti_inj': 30+273,
        'porosity': 0.046,
    })

    with open(filename, 'w') as outfile:
        json.dump(data, outfile)

## Using map() and lambda
def listOfTuples(l1, l2):
      return list(map(lambda x, y: (x, y), l1, l2))

## Plot the solution on a finite element mesh
def plot_solution(sol, outfile, title=None ):
    import plotly.figure_factory as ff
    import plotly.express as px

    p_inlet = sol[0]
    T_prod = sol[1]

    df = pd.DataFrame(listOfTuples(permeability, porosity), columns=["Permeability", "Porosity"])

    # fig = px.histogram(df, x="Permeability", y="Porosity",
    #                    marginal="box",  # or violin, rug
    #                    hover_data=df.columns)
    # fig.show()

    # sns.jointplot(x="Permeability", y="Porosity", data=df, kind="kde", n_levels=10);
    #
    # f, ax = plt.subplots(figsize=(6, 6))
    #
    # sns.kdeplot(df.Permeability, df.Porosity, n_levels=10, ax=ax)
    # sns.rugplot(df.Permeability, color="g", ax=ax)
    # sns.rugplot(df.Porosity, vertical=True, ax=ax)
    #
    # df2 = pd.DataFrame(listOfTuples(T_prod, p_inlet), columns=["T_prod", "p_inlet"])
    # print("df2", df2)
    #
    # sns.jointplot(x="T_prod", y="p_inlet", data=df2, kind="kde", n_levels=10);
    #
    # f, ax = plt.subplots(figsize=(6, 6))
    #
    # sns.kdeplot(df2.T_prod, df2.p_inlet, n_levels=10, ax=ax)
    # sns.rugplot(df2.T_prod, color="g", ax=ax)
    # sns.rugplot(df2.p_inlet, vertical=True, ax=ax)



    #Plotting
    #plt....
    #plt....
    if title:
        plt.title(title)

    #Plot configuration
    # sns.set(color_codes=True)

    #Save the figure to the output file
    plt.savefig( outfile )
    plt.show()
    print( 'Output written to {}'.format( outfile ) )