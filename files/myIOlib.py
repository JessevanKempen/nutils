#Ordering tools
import json
import numpy as np, treelog
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import collections
# import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import matplotlib.style as style
style.use('seaborn-paper')
sns.set_context("paper")
sns.set_style("whitegrid")

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
        'H': 70,
        'dtop': 2387,
        'dbasis': 2528,
        'dpump': 710,
        'dsensor': 2196,
        'labda': 0.031,
        'Tsurface': 20+273,
        'porosity': 0.2,                #'porosity': 0.046,
        'permeability': 9e-10,
        'rhof': 996.9,
        'rhos': 2400,
        'viscosity': 0.0003142,
        'K': 4e-13,
        'cpf' : 4183,
        'cps' : 2650, #870
        'labdas' : 4.2, # thermal conductivity solid [W/mK]
        'labdaf': 0.663, # thermal conductivity fluid [W/mK]
        'saltcontent': 0.155, # [kg/l]
        'pref': 225e5,
        'Tref': 90 + 273,
        'g' : 9.81,
        'rw': 0.1, #0.126
        'rmax': 1000,
        'Q': 250 / 3600,
        'L': 1000,
        'Tinj': 30 + 273,
        'patm' : 1e5,
         'ε' : 1.2,     # tubing roughness [m]
        'ct' : 1e-10,   # total compressibility

    })
    data['well'] = []
    data['well'].append({
        'Q': 250/3600,
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

def show_seaborn_plot( filename, label):
    with open(filename, 'rb') as file:
        data = np.transpose(np.load(file))

    print(data)
    df = pd.DataFrame(data[0:, 0:], columns=['f' + str(i) for i in range(data.shape[1])])
    # print(df)

    draw_df = df.reset_index().melt(id_vars=['index'], var_name='col')

    # pass custom palette:
    sns.set_palette("Spectral")
    ax = sns.lineplot(x='index',
                     y='value',
                     ci=95,
                     style=True,
                     dashes=[(2,2)],
                     legend="brief",
                     palette=("Blues_d"), #sns.color_palette('Greys')
                     hue_norm=mpl.colors.LogNorm(),
                     data=draw_df)
    plt.xlabel("time [min]", size=10)
    plt.ylabel("pressure [Pa]", size=10)
    plt.legend([label])

def show_uniform_plot():
    Qpdf = Q = np.random.uniform(low=0.1, high=1.0, size=50)

    fig, ax = plt.subplots(1, 1,
                                figsize=(10, 7),
                                tight_layout=True)

    ax.set(xlabel=r'$Q [m^3/s]$', ylabel='Probability')
    ax.hist(Q, density=True, histtype='stepfilled', alpha=0.2, bins=20)
    plt.show()

# show_seaborn_plot('pnode8.npy', "node8")
# plt.show()
