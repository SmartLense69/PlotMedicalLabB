import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as cv


def exponential(x: float | np.ndarray, A: float | np.ndarray, m: float | np.ndarray)\
        -> float | np.ndarray:
    """
    Exponential function for scipy.optimize.curvefit.
    Used to model the attenuation of gamma rays through Latex
    :param x: Distance away from the sample
    :param A: Initial count rate
    :param m: Attenuation coefficient
    :return: Resulting count rate
    """
    return A * np.exp(-m * x)


def linear(x: float | np.ndarray, m: float | np.ndarray, c: float | np.ndarray)\
        -> float | np.ndarray:
    """
    Linear function for scipy.optimize.curvefit.
    Used to model the full width half maximum (FWHM) for
    high resolution general purpose (HRGP) and
    low energy high sensitivity (LEHS) collimator
    :param x: Distance away from the sample
    :param m: Gradient of the linear fit of the FWHM
    :param c: Offset at 0cm, determined by the properties
    of the collimator
    :return: Resulting FWHM
    """
    return m * x + c


def fwhm(sigma: float | np.ndarray):
    """
    Computes the full width half maximum (FWHM) for a given
    gaussian distribution.
    The full width half maximum defines how far two
    physical objects can be before they are in-differentiable.
    :param sigma: Standard deviation of the gaussian distribution
    :return: Full width half maximum
    """
    return 2 * np.sqrt(2 * np.log(2)) * sigma


def plotInit():
    """
    Initializes the plots.
    """
    plt.rcParams['font.size'] = '30'
    plt.rcParams.update({"text.usetex": True})
    plt.figure(figsize=(9, 9))


def plotLatexVsAir():
    """
    Plots the data for latex vs air measurement data.
    Shows:

    * Measurement data for latex and air
    * Fitted exponential function for latex
    * Fitted function in a text box
    * Mean of the air measurements
    """
    plotInit()

    # Load measurements
    i1data = np.loadtxt("./results/LatexVsAir.csv", delimiter="\t", skiprows=1).transpose()

    # The error of a gaussian distribution is defined by sqrt(Number of counts)
    airError = np.sqrt(i1data[2])
    latexError = np.sqrt(i1data[1])

    # Create x-values for the fit
    xdata = np.arange(0, 24.1, 0.1)

    # popt are the fitted parameters for exponential,
    # pcov are covariances, that do not matter here
    popt, pcov = cv(exponential, i1data[0], i1data[1], p0=[35000, 1])

    # Plot error bars
    plt.errorbar(i1data[0], i1data[2], xerr=0.05, yerr=airError, label="Air",
                 marker='.', ecolor='r', ls='none', color='r', capsize=3)
    plt.errorbar(i1data[0], i1data[1], xerr=0.05, yerr=latexError, label="Latex",
                 marker='.', ecolor='b', ls='none', color='b', capsize=3)

    # Display mean of air to show how constant it is
    plt.axhline(np.mean(i1data[2]), color='r', linestyle='dashed', label="Mean of air")

    # Display fitted function
    plt.plot(xdata, exponential(xdata, popt[0], popt[1]), color='b', label='Fit for latex',
             linestyle='dashed')

    # Create a text box with the fitted function inside
    fitLabel = "$f(x)=" + str(round(popt[0], 2)) + "\\times e^{-" + str(round(popt[1], 2)) + "x}$"
    plt.text(9, 260000, fitLabel, bbox=dict(facecolor='white', alpha=1))

    # Change ticks so the measurement data is on grid lines
    plt.xticks(np.arange(0, 28, 4))

    plt.grid()
    plt.legend()
    plt.xlabel("Distance in cm")
    plt.ylabel("Total gamma counts")
    plt.xlim(left=0)
    plt.ylim(bottom=0)

    # Add coordinate axes in gray
    plt.axvline(0, color='gray')
    plt.axhline(0, color='gray')

    # To make the data more comparable between graphs, set magnitude of the counts to 10^(5).
    plt.ticklabel_format(axis='y', style='sci', scilimits=(5, 5))
    plt.show()


def plotCountVsActivity():
    """
    Makes the plot for count rate over different activities.
    Shows:
    
    * Measurement data
    * Box for all potential false negatives
    """
    plotInit()
    i2data = np.loadtxt("./results/VariousActivity.csv", delimiter="\t", skiprows=1).transpose()
    i2data[1] = i2data[1] / 10
    i2err = np.sqrt(i2data[1])

    # Creates a hatched box in the area of false negatives
    plt.axhspan(i2data[1, -1], np.max(i2data[1]), alpha=0.5, color='r', linestyle='dashed',
                lw=2, fill=False, hatch='/', label="Potential false negatives")

    plt.errorbar(i2data[0], i2data[1], xerr=0.1, yerr=i2err, marker='x',
                 label="Gamma counts", color='r', capsize=3, ls='none', ecolor='r')
    plt.xlabel("Activity in MBq")
    plt.ylabel("Gamma counts in $s^{-1}$")
    plt.axvline(0, color='gray')
    plt.axhline(0, color='gray')
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.legend()
    plt.grid()
    plt.ticklabel_format(axis='y', style='sci', scilimits=(5, 5))
    plt.show()


def plotFWHM():
    """
    Shows the FWHM for HRGP and LEHS
    Plots:

    * Measurement data for HRGP and LEHS
    * Fitted linear function for HRGP and LEHS
    * Text-boxes for fitted function of HRGP and LEHS
    """
    plotInit()
    hrgp = np.loadtxt("./results/HRGP.csv", delimiter="\t", skiprows=1).transpose()
    lehs = np.loadtxt("./results/LEHS.csv", delimiter="\t", skiprows=1).transpose()
    xdata = np.arange(np.min(hrgp[0]) - 10, np.max(hrgp[0]), 0.1)
    poptHGRP, pcov = cv(linear, hrgp[0], fwhm(hrgp[2]), p0=[1, 1])
    poptLEHS, pcov = cv(linear, lehs[0], fwhm(lehs[2]), p0=[1, 1])
    plt.grid()
    plt.scatter(hrgp[0], fwhm(hrgp[2]), color='r', marker='o', label="HRGP")
    plt.scatter(lehs[0], fwhm(lehs[2]), color='b', marker='v', label="LEHS")
    plt.plot(xdata, linear(xdata, poptHGRP[0], poptHGRP[1]), color='r', label='Fit HRGP',
             linestyle='dashed')
    fitLabelHRGP = "HRGP$(x)=" + str(round(poptHGRP[0], 2)) + "\\times x + " + str(round(poptHGRP[1], 2)) + "$"
    fitLabelLEHS = "LEHS$(x)=" + str(round(poptLEHS[0], 2)) + "\\times x + " + str(round(poptLEHS[1], 2)) + "$"
    plt.text(10, 19.5, fitLabelLEHS, bbox=dict(facecolor='white', alpha=1))
    plt.plot(xdata, linear(xdata, poptLEHS[0], poptLEHS[1]), color='b', label='Fit LEHS',
             linestyle='dashed')
    plt.text(10, 5.5, fitLabelHRGP, bbox=dict(facecolor='white', alpha=1))
    plt.xlabel("Distance in cm")
    plt.ylabel("Full width half maximum in cm")
    plt.axvline(0, color='gray')
    plt.axhline(0, color='gray')
    plt.xlim(left=-5)
    plt.ylim(bottom=0)
    plt.legend()
    plt.show()


def plotFirstKidney():
    """
    Makes the plot for the first set of kidneys.
    See plotSecondKidney for the second set.
    Shows:

    * Activity for the left and right kidney
    * LKF and RKF
    """
    plotInit()
    i41data = np.loadtxt("./results/MAGN-1.csv", delimiter="\t", skiprows=1).transpose()
    xdata = np.arange(0, 80, 1)

    # LKCC/RKCC is the integral over all data values, i.e.
    # the sum of all datapoints
    LKCC = np.sum(i41data[0])
    RKCC = np.sum(i41data[1])
    LKF = LKCC / (LKCC + RKCC)
    RKF = RKCC / (LKCC + RKCC)
    print("LKF:")
    print(LKF)
    print("RKF:")
    print(RKF)

    plt.scatter(xdata, i41data[0], 50, color='r', marker='o', label="Left Kidney")
    plt.scatter(xdata, i41data[1], 50, color='b', marker='v', label="Right Kidney")
    plt.xlabel("Time in minutes")
    plt.ylabel("Counts")
    plt.axvline(0, color='gray')
    plt.axhline(0, color='gray')
    plt.xlim(left=0, right=80)
    plt.ylim(bottom=0)
    plt.grid()
    plt.legend()
    plt.ticklabel_format(axis='y', style='sci', scilimits=(5, 5))
    plt.show()


def plotSecondKidney():
    """
    Similar to plotFirstKidney(), but now for the second set of kidneys
    """
    plotInit()
    i42data = np.loadtxt("./results/MAG30-2.csv", delimiter="\t", skiprows=1).transpose()
    xdata = np.arange(0, 80, 1)

    LKCC = np.sum(i42data[0])
    RKCC = np.sum(i42data[1])
    LKF = LKCC / (LKCC + RKCC)
    RKF = RKCC / (LKCC + RKCC)
    print("LKF:")
    print(LKF)
    print("RKF:")
    print(RKF)

    plt.scatter(xdata, i42data[0], 50, color='r', marker='o', label="Left Kidney")
    plt.scatter(xdata, i42data[1], 50, color='b', marker='v', label="Right Kidney")
    plt.xlabel("Time in minutes")
    plt.ylabel("Counts")
    plt.axvline(0, color='gray')
    plt.axhline(0, color='gray')
    plt.xlim(left=0, right=80)
    plt.ylim(bottom=0)
    plt.grid()
    plt.legend()
    plt.ticklabel_format(axis='y', style='sci', scilimits=(5, 5))
    plt.show()


if __name__ == '__main__':
    plotLatexVsAir()
    plotCountVsActivity()
    plotFWHM()
    plotFirstKidney()
    plotSecondKidney()
