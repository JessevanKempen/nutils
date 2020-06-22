import json
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import lognorm
from scipy import stats
import math
import pandas as pd
# import seaborn as sns
from myUQ import *


## Parameters txt file reader
def read_from_txt ( filename ):

    # Open the parameter file to read
    with open(filename, 'r') as json_file:
            data = json.load(json_file)

    # for key, value in data['aquifer'][0].items():
        # print(key, ":", value)

    # for key, value in data['well'][0].items():
        # print(key, ":", value)

    return data['aquifer'][0], data['well'][0]

        # for key, value in aquifer.items():
        #     self.parameter = value
        #     print("parameter", self.parameter)
            # print(key, ":", value)

        # aquifer characteristics
        # self.key = value
        # print("testtest", self.key)

    # for x in data['aquifer']:
    #     print("x",x[1])
        # print("%s: %d" % (x, x[x]))
    # print(well)

    # for aq in data['aquifer']:
    #     d_top = aq['d_top']  # depth top aquifer at production well
    #     labda = aq['labda']  # geothermal gradient
    #     reservoir.H = aq['H']  # thickness aquifer
    #     T_surface = aq['T_surface']
    #     porosity = aq['porosity']
    #     rho_f = aq['rho_f']
    #     mhu = aq['mhu']
    #     K = aq['K']
    #     print(H)

            #Open the parameter file to read
    # open( parameters.txt, "r")
    # open( "fname", "r" )
    # open("fname.txt", "r")

    # with open('parameters.txt') as json_file:
    #     data = json.load(json_file)
    #     for aq in data['aquifer']:
    #         self.d_top = aq['d_top']  # depth top aquifer at production well
    #         self.labda = aq['labda']  # geothermal gradient
    #         self.H = aq['H']  # thickness aquifer
    #         self.T_surface = aq['T_surface']
    #         self.porosity = aq['porosity']
    #         self.rho_f = aq['rho_f']
    #         self.mhu = aq['mhu']
    #         self.K = aq['K']

from matplotlib import pyplot as plt


## Generates a Parameters txt file
def generate_txt( filename):
    import json

    data = {}
    data['aquifer'] = []
    data['aquifer'].append({
        'd_top': 2400,
        'labda': 0.031,
        'H': 140,
        'T_surface': 20,
        'porosity': 0.05,
        'permeability': 9e-10,
        'rho_f': 1080,
        'rho_s': 2711,
        'viscosity': 8.9e-4,
        'K': 4e-13,
        'Cp_f' : 4183,
        'Cp_s' : 910,
        'labda_s' : 1.9,
        'g' : 9.81
    })
    data['well'] = []
    data['well'].append({
        'r': 0.126, #0.076,
        'Q': 330/3600,
        'L': 1000,
        'Ti_inj': 30,
        'epsilon': 0.046,
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