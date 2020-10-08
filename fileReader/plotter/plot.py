from matplotlib import pyplot as plt
import numpy as np
import math
from matplotlib import rc
from matplotlib.ticker import FormatStrFormatter


def create_plots(dist_data, speedups):
    def get_y_ticks(data, num_points=6):
        data = list(filter(lambda x: x>0.0, data))
        lst = list(map(lambda x: truncate(x, 2) if x > 0.01 else x, np.geomspace(min(data), max(data), num_points)))
        lst.append(1.00)
        return sorted(lst)

    def truncate(number, decimals=0):
        """
        Returns a value truncated to a specific number of decimal places.
        """
        if not isinstance(decimals, int):
            raise TypeError("decimal places must be an integer.")
        elif decimals < 0:
            raise ValueError("decimal places has to be 0 or more.")
        elif decimals == 0:
            return math.trunc(number)

        factor = 10.0 ** decimals
        return math.trunc(number * factor) / factor

    fig = plt.figure()
    plt.style.use('bmh')

    ### Subplot A ###

    ax = fig.add_subplot(121)
    plt.text(0.5, 1.05, "(a)", transform=ax.transAxes)
    ys = []
    for algorithm in speedups:
        for machine in speedups[algorithm]:
            ys += speedups[algorithm][machine][1]
            plt.plot(np.arange(len(speedups[algorithm][machine][0])), speedups[algorithm][machine][1], label=str(algorithm) + "-" + str(machine))

    plt.yscale('log')
    yticks = get_y_ticks(ys)
    plt.yticks(yticks, yticks)
    plt.tick_params(which='minor', left=False)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.legend(fancybox=True, framealpha=0.5)

    ### Subplot B ###

    ax2 = fig.add_subplot(122)
    plt.text(0.5, 1.05, "(b)", transform=ax2.transAxes)

    #create x and y axis arrays
    ys = []
    for algorithm in dist_data:
        for machine in dist_data[algorithm]:
            ys += dist_data[algorithm][machine]
            plt.plot(np.arange(len(dist_data[algorithm][machine])), dist_data[algorithm][machine], label=str(algorithm) + "-" + str(machine))

    plt.legend(fancybox=True, framealpha=0.5)
    plt.yscale('log')
    yticks= get_y_ticks(ys, 7)
    plt.yticks(yticks, yticks)
    plt.tick_params(which='minor', left=False)

    rc('font', ** {'family':'serif', 'serif':['Times']})
    rc('text', usetex = True)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.show()





