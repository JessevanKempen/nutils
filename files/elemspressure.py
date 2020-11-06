# plot elems size: 50, 100, 200, 400
# plot FEA and EX pressure
# use t=600s
from nutils import mesh, function, solver, export, cli, topology, sparse, types
import numpy as np
import math

def main():

    length = 1000
    selems = [length/25,
              length/50,
              length/100,
              length/200,
              length/400,
              length/1000]
    nelems = [2.5,
              5,
              10,
              20,
              40,
              100,
              200] #elements per 100m
    print(selems)

    #t=600, L=1000
    pFEA = [22086111.42748884,
            21902014.957589053,
            21726777.721221756,
            21562538.447041728, #dz = 5m
            21411662.357217178,
            21243415.78431648]
    pEX = [20700173.2868534,
           20700173.2868534,
           20700173.2868534,
           20700173.2868534, #dz = 5m
           20700173.2868534,
           20700173.2868534]
    print(pFEA, pEX)

    # t=600, L=250
    pFEA2 = [22145699.217542585,
            (21912576.577329777+21891746.037332077)/2,
            21726704.23802078,  #dz = 10m
            21562470.026213087, #dz = 5m
            21411600.745572496, #dz = 2.5m
            21243475.78431648]  #dz = 1m

    pEX2 =  [20700173.2868534,
            20700173.2868534,
            20700173.2868534,
            20700173.2868534,   #dz = 5m
            20700173.2868534,   #dz = 2.5m
            20700173.2868534]   #dz = 1m



    pFEA2refined = [21135143.533079684,
                    21103572.526394643, #20m (5/100), (12.5 element op 250)
                    21068682.36205357, #10m (10/100), (25 elements)
                    21055465.52883131] #1m (100/100m), (250elements op L=250)
    nelemsrefined = [2.5,
                     5,
                     10,
                     100]
    pEX2refined =  [20700173.2868534,
                    20700173.2868534,
                    20700173.2868534,
                    20700173.2868534]

    # t=600 L=250
    dpFEA = [-1305968.4288678593, #dz = 20m
             -2921755.772435982,
             -5885792.714165609,
             -11436990.774427539, #dz = 2.5m
             -25816941.087984808,
             -44298204.62592584] #dz = 0.5m
    dpEX = [-35996534.26293199, #dz = 20m
            -35996534.26293199,
            -35996534.26293199,
            -35996534.26293199,
            -35996534.26293199,
            -35996534.26293199]

    #calculate L2 and H1 error, ns.du_i = 'u_i - uexact_i'
    dp3 = np.subtract(pFEA2refined, pEX2refined)

    dp2 = np.subtract(pFEA2, pEX2)
    dp = np.subtract(pFEA, pEX)
    print("pressure difference", dp2)
    l2error = (dp2*np.transpose(dp2))**0.5
    print("l2error", l2error)

    dpdp = np.subtract(dpFEA, dpEX)
    print("gradient difference", dpdp)
    H1error = (dp2*np.transpose(dp2) + dpdp*np.transpose(dpdp))**0.5
    print("H1error", H1error)



    # with export.mplfigure('elemspressure.png', dpi=800) as plt:
    #     ax1 = plt.subplots()
    #     ax1.set(xlabel='dz [m]')
    #     ax1.invert_xaxis()
    #     ax1.set_ylabel('Pressure [Pa]')
    #     ax1.plot(selems, pFEA, 'b--', label="FEA R=1000m")
    #     ax1.plot(selems, pEX2, label="analytical")
    #     ax1.plot(selems, pFEA2, '--', label="FEA R=250m")
    #     ax1.legend(loc="center right")
    #
    # with export.mplfigure('error.png', dpi=800) as plt:
    #     ax1 = plt.subplots()
    #     ax1.set(xlabel='dz [m]')
    #     ax1.invert_xaxis()
    #     ax1.set_ylabel('Difference [-]')
    #     ax1.plot(selems, dp/225e5, label="error")
    #     ax1.legend(loc="center right")

    #number of elements per 100m
    #log plot
    # with export.mplfigure('relativeerrorL2.png', dpi=800) as plt:
    #     # plt.add_axes([0.6, 0.3, 0.3, 0.3])
    #     ax1 = plt.subplots()
    #     ax1.set(xlabel=r'$N_{e}$ per 100m', xscale="log")  #'r''
    #     ax1.set(ylabel=r'$\left(\left|p_{w}-{p}_{w,exact}\right|/\left|p_{w,0}\right|\right)$', yscale="log")
    #     ax1.plot(nelems, dp2/225e5, 'o--', color='orange', label=r'$r_{dr} = 250m$ uniform mesh')
    #     ax1.plot(nelems, dp/225e5, 'ob--', label=r'$r_{dr} = 1000m$ uniform mesh')
    #     ax1.plot(nelemsrefined, dp3/ 225e5, 'o--', label=r'$r_{dr} = 250m$ refined mesh')
    #     ax1.legend(loc="upper right")
    #     plt.tight_layout()
    #
    # with export.mplfigure('errorL2.png', dpi=800) as plt:
    #     # plt.add_axes([0.6, 0.3, 0.3, 0.3])
    #     ax1 = plt.subplots()
    #     ax1.set(xlabel=r'$N_{e}$ per 100m', xscale="log")  # 'r''
    #     ax1.set(ylabel=r'$\left|p_{w}-{p}_{w,exact}\right|_{L^{2}-norm}$', yscale="log")
    #     ax1.plot(nelems, l2error, 'o--', color='red', label=r'$r_{dr} = 250m$, uniform mesh')
    #     ax1.legend(loc="upper right")
    #     plt.tight_layout()

    with export.mplfigure('errorH1.png', dpi=800) as plt:
        # plt.add_axes([0.6, 0.3, 0.3, 0.3])
        ax1 = plt.subplots()
        ax1.set(xlabel=r'$N_{e}$ per 100m', xscale="log")  # 'r''
        ax1.set(ylabel=r'$\sqrt{\left|(p_{w}-{p}_{w,exact})^2+(p_{w}^{\prime}-{p}_{w,exact}^{\prime})2\right|}{H^{1}-norm}$', yscale="log")
        ax1.plot(nelems[1:], H1error, 'o--', color='red', label=r'$r_{dr} = 250m$, uniform mesh')
        ax1.legend(loc="upper right")
        plt.tight_layout()

if __name__ == '__main__':
    cli.run(main)
