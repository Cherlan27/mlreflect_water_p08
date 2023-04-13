"""
2d_display.py - Display two-dimensional data.
Use this script to plot two-dimensional data obtained with the Eiger,
Pilatus or Lambda detector and visualize regions of interest.
"""
import h5py
import numpy
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Rectangle
import hashlib
import os
from PIL import Image
from eigene.p08_detector_read import p08_detector_read




# == settings
settings = dict(
#                data_directory = "C:/Users/hoevelmann/Desktop/Beamtime_timing/raw",
 #               data_directory = "C:/Users/hoevelmann/Desktop/Beamtime_Rennie/raw",
                flatfield = "./Module_2017-004_GaAs_MoFluor_Flatfielddata.tif",
                pixel_mask = "./Module_2017-004_GaAs_mask.tif",
                use_flatfield = True,
                data_directory = "K:/SYNCHROTRON/Murphy/2022-04_P08_11013326_giri_hoevelmann_petersdorf/raw_data/raw", #"K:/SYNCHROTRON/Murphy/2021-10_P08_11010876_giri_hoevelmann_petersdorf/raw_data/raw", #"./raw", #"raw", # "K:/SYNCHROTRON/Murphy/2021-10_P08_11010876_giri_hoevelmann_petersdorf/raw_data/raw", #"./raw",
                #experiment = "align_18keV_2",
                scan_number = 1350,
                experiment = "nacl",
                detector = "lambda",
 #               scan_number = 195,
                image_number =  1, # 27 - 31 - 34
                log_scale = True,
                intensity_scale = (1, 100),
                # plot colormap
                colormap = "viridis",
                # Specify which data to use for x and y dimensios. Supported
                # values are "pixels", "qx", "qy" and "qz". If reciprocal space
                # data is not found, "pixels" will be used.
                xdim = "pixels",
                ydim = "pixels",
                # regions of interest to display (x, y, w, h)
                rois = {#"roi": (1470,360,60,20),
                        "roi": (14, 91, 60, 22),
                        #"roi": (35, 85, 22, 15),
                       # "roi": (1480,390,70,30),
                       # "roi2_old": (14,68,50,22),
                        #"roi1": (1485,180,60,22),
                        #"roi_1": (140,85,60,22),
                        #"roi": (1450,354,90,20),
#                        "roi_2": (50,480,11,21),
#                        "roi1": (721, 283, 10, 10),
#                        "roi2": (515, 400, 1, 1),
#                        "bkg10": (701, 283+10+20, 50, 10),
#                        "bkg11": (701, 283-10-20, 50, 10)
                        },
                # save plot to this file:
                save_plot = None,
                plot_out_filename = "./processed/test.png",
                # plot title
                plot_title="A Diffraction Measurement"
                )
                
AXIS_LABELS = dict(pixels="pixels", qx=r"q$_x$ [\AA$^{-1}$]",
                   qy=r"q$_y$ [\AA$^{-1}$]", qz=r"q$_z$ [\AA$^{-1}$]")


# == load data
#det_filename = "{data_directory}/{user_group}/{user_run}/{experiment}/{detector}/nexus/{user_group}_{user_run}_{experiment}_scan{scan_number:06}.nx5".format(**settings)
#det_file = h5py.File(det_filename, "r")
#img = numpy.array(det_file["/scan/lisa/{0}/counts".format(settings["detector"])][settings["image_number"]])
img = p08_detector_read(settings["data_directory"], settings["experiment"], settings["scan_number"], settings["detector"])()[settings["image_number"]]
Q = None
#if "Q" in det_file["/scan/lisa/{0}".format(settings["detector"])]:
#    Q = numpy.array(det_file["/scan/lisa/{0}/Q".format(settings["detector"])][settings["image_number"]])

# if settings["use_flatfield"] == True:
#     flatfield_2 = numpy.ones((516,1556))
#     flatfield = numpy.array(Image.open(settings["flatfield"]))
#     for i in range(len(flatfield)):
#         for j in range(len(flatfield[i])):
#             flatfield_2[515-i][1553-j] = flatfield[i][j]
#     flatfield = flatfield_2

# == flatfield correction
if settings["use_flatfield"] == True:
    flatfield = numpy.array(Image.open(settings["flatfield"]))
    img = img / flatfield

# == plot
rcParams["figure.figsize"] = 8, 5
rcParams["font.size"] = 20
rcParams["text.usetex"] = False
rcParams["text.latex.preamble"] = r"\usepackage{sfmath}"
fig = plt.figure(dpi = 600)
fig.patch.set_color("white")
ax = fig.gca()

smin, smax = settings["intensity_scale"]

if settings["log_scale"]:
    data = numpy.log10(img)
    vmin = max(numpy.log10(smin), 0)
    vmax = numpy.log10(smax)
else:
    data = img
    vmin, vmax = smin, smax

# make xdata
if settings["xdim"] == "pixels" or Q is None:
    xdata, _ = numpy.meshgrid(numpy.arange(img.shape[1]),
                              numpy.arange(img.shape[0]))
elif settings["xdim"] == "qx":
    xdata = Q[:,:,0] * 1e-10
elif settings["xdim"] == "qy":
    xdata = Q[:,:,1] * 1e-10
elif settings["xdim"] == "qz":
    xdata = Q[:,:,2] * 1e-10
# make ydata                            
if settings["ydim"] == "pixels" or Q is None:
    _, ydata = numpy.meshgrid(numpy.arange(img.shape[1]),
                              numpy.arange(img.shape[0]))
elif settings["ydim"] == "qx":
    ydata = Q[:,:,0] * 1e-10
elif settings["ydim"] == "qy":
    ydata = Q[:,:,1] * 1e-10
elif settings["ydim"] == "qz":
    ydata = Q[:,:,2] * 1e-10
# plot data
ax.pcolormesh(xdata, ydata, data, cmap=settings["colormap"], 
          vmin=vmin, vmax=vmax)
# plot rois
for roi_name in settings["rois"]:
    roi = settings["rois"][roi_name]
    
    color = "#" + hashlib.md5(roi_name.encode('utf-8')).hexdigest()[:6]
    # top line
    x = xdata[roi[1], roi[0]:(roi[0]+roi[2]+1)]
    y = ydata[roi[1], roi[0]:(roi[0]+roi[2]+1)]
    ax.plot(x, y, color=color)
    # bottom line
    x = xdata[roi[1]+roi[3], roi[0]:(roi[0]+roi[2]+1)]
    y = ydata[roi[1]+roi[3], roi[0]:(roi[0]+roi[2]+1)]
    ax.plot(x, y, color=color)
    # left line
    x = xdata[roi[1]:(roi[1]+roi[3]+1), roi[0]]
    y = ydata[roi[1]:(roi[1]+roi[3]+1), roi[0]]
    ax.plot(x, y, color=color)
    # right line
    x = xdata[roi[1]:(roi[1]+roi[3]+1), roi[0]+roi[2]]
    y = ydata[roi[1]:(roi[1]+roi[3]+1), roi[0]+roi[2]]
    ax.plot(x, y, color=color)
    # roi name
    ax.annotate(roi_name, (xdata[roi[1], roi[0]], ydata[roi[1], roi[0]]),
                color=color, ha="left", va="bottom", size=12)




# configure plot
if settings["xdim"] == "pixels" or Q is None:
    ax.xaxis.set_minor_locator(MultipleLocator(20))
if settings["ydim"] == "pixels" or Q is None:
    #ax.set_ylim(img.shape[0], 0)
    ax.yaxis.set_minor_locator(MultipleLocator(20))
ax.set_aspect("auto")
ax.set_xlabel(AXIS_LABELS[settings["xdim"]] if Q is not None else "x-pixel", fontsize = 20)
ax.set_ylabel(AXIS_LABELS[settings["ydim"]] if Q is not None else "y-pixel", fontsize = 20)

# plt.xticks(labelsize = 20)
# plt.yticks(labelsize = 20)

#ax.set_xlim([0,100])
#ax.set_ylim([70,110])

ax.set_xlim([0,100])
ax.set_ylim([50,210])
# if settings["plot_title"] is not None:
#     ax.set_title(settings["plot_title"])
plt.subplots_adjust(bottom=0.12)

if settings["save_plot"]:
    if os.path.exists(settings["plot_out_filename"]):
        settings["save_plot"] = input("Plot output picture file already exists. Overwrite? [y/n] ") == "y"
    if settings["save_plot"]:
        plt.savefig(settings["plot_out_filename"], dpi=300)

plt.show()
print()