from eigene.fio_reader import read
import numpy
import h5py
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator
import pandas
import os
from eigene.abs_overlap_fit_poly import Absorber
import sys
from PIL import Image
from eigene.p08_detector_read import p08_detector_read
from normalization_ref import normalisator




plt.ioff()
#plt.close("all")

"""
Use this script to create reflectivity curves from several Spock q scans.
This only works with NeXus data from the Eiger, Pilatus or
Lambda detectors.
"""


# == settings
settings =  dict(
                data_directory = "./raw",
                flatfield = "./Module_2017-004_GaAs_MoFluor_Flatfielddata.tif",
                pixel_mask = "./Module_2017-004_GaAs_mask.tif",
                use_flatfield = True,
                use_mask = None,
                experiment = "timing",
                detector = "lambda",
                scan_numbers = list(range(1803,1811+1)),
                nom_scan_numbers = list(range(1803,1811+1)),
                detector_orientation = "vertical",
                footprint_correct = True,
                beam_width = 20e-6, #2e-5, 0.2e-4, 0.02e-3,
                sample_length = 81.4e-3,
                # x, y, w, h of specular ROI:
                roi = (14,85,60,22),# (1506, 185, 18, 15)
                # offset of background ROIs:
                roi_offset = 30,
                #monitor:
                monitor = "ion2",
                # absorber factors:
                calculate_abs = True,           # If "True" the absorber factors wíll be calculated and the table will be ignored
                absorber_factors = {1: 12.617,
                                    2: 11.0553,
                                    3: 11.063,
                                    4: 11.048,
                                    5: 11.7,
                                    6: 12.813},
                primary_intensity = "normalized",     # use "auto" for the calculation between auto_cutoff[0] and qc-auto_cutoff[1], use "scan" for the calculation from a primary beam scan, use an integer for a manually intensity 
                auto_cutoff = [0.015, 0.003],
                auto_cutoff_nom = [0.0184, 0.0013],
                scan_number_primery = 1137,
                qc = 0.025,    # rbbr 5 mol  
                # roughness for fresnel plot:
                roughness = 2.55, 
                # output data:
                save_results = True,
                show_plot = True,
                save_plot = True,
                # results are written to this file:
                out_data_directory = "./processed/",
                out_user_run = "2021-04",
                out_experiment = "rbbr",
                out_typ_experiment = "test_5mol", 
                out_state = "",
                out_settings = "",
                # plot title
                plot_title=""
                )
    
# == helper functions
def abs_fac(abs_val):
    abs_val = int(abs_val + 0.2)
    if abs_val == 0:
        return 1.0
    else:
        return settings["absorber_factors"][abs_val] * abs_fac(abs_val - 1)
        

def fresnel(qc, qz, roughness=2.5):
    """
    Calculate the Fresnel curve for critical q value qc on qz assuming
    roughness.
    """
    return (numpy.exp(-qz**2 * roughness**2) *
            abs((qz - numpy.sqrt((qz**2 - qc**2) + 0j)) /
            (qz + numpy.sqrt((qz**2 - qc**2)+0j)))**2)

# wav = wavelength
# L = sample width
# b = width of the beam
def footprint_correction(q, intensity, b, L, wl = 68.88e-12):
    """ 
    calculates the footprint corrected data
    """
    q_b = (4*numpy.pi/wl*b/L)*10**(-10)
    print(q_b)
    intensity2 = intensity
    print(q[len(q)-1] > q_b)
    i = 0
    for i in range(0,len(q),1):
        if q[i] < q_b:
            print(i)
            print(q[i])
            intensity2[i] = intensity2[i]/(q[i]/q_b)
            i += 1
    else:
        None
    return intensity2, q_b

if settings["use_flatfield"] == True:
    flatfield_2 = numpy.ones((516,1556))
    flatfield = numpy.array(Image.open(settings["flatfield"]))


if settings["use_mask"]:
    file_mask = h5py.File(settings["mask"], "r")
    img_mask = numpy.array(file_mask["/entry/instrument/detector/data"])[0]
    file_mask.close()
    mask = numpy.zeros_like(img_mask)
    mask[(img_mask > 1)] = 1
    mask = (mask == 1)
    mask_value = 0


# == prepare data structures
intensity = numpy.array([])
e_intensity = numpy.array([])
qz = numpy.array([])

# == make data
absorbers = Absorber()
temp_intens = {}
temp_e_intens = {}

for scan_number in settings["scan_numbers"]:
# == load .fio file
    fio_filename = "{data_directory}/{experiment}_{0:05}.fio".format(scan_number, **settings)
    header, column_names, data, scan_cmd = read(fio_filename)
    # load monitor
    s_moni = data[settings["monitor"]]
    # make qz
    wl = 12.38/18 * 1e-10
    s_alpha = data["alpha_pos"]
    s_beta = data["beta_pos"]
    s_qz = ((4 * numpy.pi / wl) *
            numpy.sin(numpy.radians(s_alpha + s_beta) / 2.0) * 1e-10)
    # prepare data structures
    s_intens = []
    s_e_intens = []
    # load detector data  
    roi = settings["roi"]
    roi_offset = settings["roi_offset"]
    detector_images = p08_detector_read(settings["data_directory"], settings["experiment"], scan_number, settings["detector"])()
    n_images = detector_images.shape[0]
    n_points = min(n_images, len(s_alpha))
    
    for n in range(n_points):
        img = detector_images[n]
        
        # flatfield correction
        if settings["use_flatfield"] == True:
            img = img / flatfield
        
        if settings["use_mask"] == True:
            img[mask] = mask_value
        
        p_specular = img[roi[1]:(roi[1]+roi[3]),roi[0]:(roi[0]+roi[2])].sum()            

        if settings["detector_orientation"] == "horizontal":            
            p_bg0 = img[roi[1]:(roi[1]+roi[3]),
                        (roi[0]+roi[2]+roi_offset):(roi[0]+2*roi[2]+roi_offset)].sum()
            p_bg1 = img[roi[1]:(roi[1]+roi[3]),
                        (roi[0]-roi[2]-roi_offset):(roi[0]-roi_offset)].sum()            
        elif settings["detector_orientation"] == "vertical":            
            p_bg0 = img[(roi[1]+roi[3]+roi_offset):(roi[1]+2*roi[3]+roi_offset),
                        (roi[0]):(roi[0]+roi[2])].sum()
            p_bg1 = img[(roi[1]-roi[3]-roi_offset):(roi[1]-roi_offset),
                        (roi[0]):(roi[0]+roi[2])].sum()       
            
        p_intens = ((p_specular - (p_bg0 + p_bg1) / 2.0) / s_moni[n])
        
        if settings["monitor"] == "Seconds":
            p_e_intens = ((numpy.sqrt(p_specular) + (numpy.sqrt(p_bg0) + numpy.sqrt(p_bg1)) / 2.0) / s_moni[n])
        else:    
            p_e_intens = ((numpy.sqrt(p_specular) + (numpy.sqrt(p_bg0) + numpy.sqrt(p_bg1)) / 2.0) / s_moni[n] 
                         + abs (0.1 * (p_specular - (p_bg0 + p_bg1) / 2.0) / s_moni[n]))
        s_intens.append(p_intens)
        s_e_intens.append(p_e_intens)
    s_intens = numpy.array(s_intens)
    s_e_intens = numpy.array(s_e_intens)
    
    qz = numpy.concatenate((qz, s_qz))
    if settings["footprint_correct"] == True:
            temp_intensities = s_intens
            proof = s_intens
            print(s_qz)
            temp_intensities, q_b = footprint_correction(s_qz, temp_intensities, settings["beam_width"], settings["sample_length"])
            s_intens = temp_intensities
    
    if settings["calculate_abs"] == True:
        absorbers.add_dataset(header["abs"], s_qz, s_intens)
        temp_intens[scan_number] = int(header["abs"]+0.1), s_intens
        temp_e_intens[scan_number] = int(header["abs"]+0.1), s_e_intens
    elif settings["calculate_abs"] == None:
        s_intens = s_intens * abs_fac(header["abs"])
        s_e_intens = s_e_intens * abs_fac(header["abs"])
        intensity = numpy.concatenate((intensity, s_intens))
        e_intensity = numpy.concatenate((e_intensity, s_e_intens))
    else:
        sys.exit("Bitte Absorberfaktoren spezifizieren")
    
if settings["calculate_abs"] == True:
    absorbers.calculate_from_overlaps()
    intensity = numpy.concatenate([absorbers(x[0])*x[1] for x in list(temp_intens.values())])
    e_intensity = numpy.concatenate([absorbers(x[0])*x[1] for x in list(temp_e_intens.values())])
    
# == sort data points
m = numpy.argsort(qz)
qz = qz[m]
intensity = intensity[m]
e_intensity = e_intensity[m]
    
# == normalize
if settings["primary_intensity"] == "auto":
    primary = intensity[(qz > settings["auto_cutoff"][0]) & (qz<(settings["qc"] - settings["auto_cutoff"][1]))].mean()
elif settings["primary_intensity"] == "scan":
    # data for normalization 
    # load scan
    fio_filename = "{data_directory}/{experiment}_{scan_number_primery:05}.fio".format(**settings)
    header, column_names, data, scan = read(fio_filename)
    # load monitor
    s_moni = data[settings["monitor"]]
    # make qz
    wl = 12.38/18 * 1e-10 / 1.6e-19
    s_alpha = data["alpha_pos"]
    s_beta = data["beta_pos"]
    s_qz = ((4 * numpy.pi / wl) *
            numpy.sin(numpy.radians(s_alpha + s_beta) / 2.0) * 1e-10)
    # prepare data structures
    s_intens = []
    s_e_intens = []
    # load detector data
    roi = settings["roi"]
    roi_offset = settings["roi_offset"]
    detector_images = p08_detector_read(settings["data_directory"], settings["experiment"], settings["scan_number_primery"], settings["detector"])()
    n_images = detector_images.shape[0]
    
    img = detector_images[0]
    p_specular = img[roi[1]:(roi[1]+roi[3]),roi[0]:(roi[0]+roi[2])].sum()
    
    if settings["detector_orientation"] == "horizontal":            
        p_bg0 = img[roi[1]:(roi[1]+roi[3]),
                    (roi[0]+roi[2]+roi_offset):(roi[0]+2*roi[2]+roi_offset)].sum()
        p_bg1 = img[roi[1]:(roi[1]+roi[3]),
                    (roi[0]-roi[2]-roi_offset):(roi[0]-roi_offset)].sum()            
    elif settings["detector_orientation"] == "vertical":            
        p_bg0 = img[(roi[1]+roi[3]+roi_offset):(roi[1]+2*roi[3]+roi_offset),
                    (roi[0]):(roi[0]+roi[2])].sum()
        p_bg1 = img[(roi[1]-roi[3]-roi_offset):(roi[1]-roi_offset),
                    (roi[0]):(roi[0]+roi[2])].sum()
                    
    p_intens = ((p_specular - (p_bg0 + p_bg1) / 2.0) / s_moni[0])
    
    if settings["monitor"] == "Seconds":
        p_e_intens = ((numpy.sqrt(p_specular) + (numpy.sqrt(p_bg0) + numpy.sqrt(p_bg1)) / 2.0) / s_moni[0])
    else:    
        p_e_intens = ((numpy.sqrt(p_specular) + (numpy.sqrt(p_bg0) + numpy.sqrt(p_bg1)) / 2.0) / s_moni[0] 
                     + abs (0.1 * (p_specular - (p_bg0 + p_bg1) / 2.0) / s_moni[0]))
    
    s_intens.append(p_intens)
    s_e_intens.append(p_e_intens)
    norm_intens = numpy.array(s_intens)*absorbers(list(temp_e_intens.values())[0][0])
    norm_e_intens = numpy.array(s_e_intens)*absorbers(list(temp_e_intens.values())[0][0])
    primary = norm_intens[0]
elif settings["primary_intensity"] == "normalized":
    primary_ref = intensity[(qz > settings["auto_cutoff"][0]) & (qz<(settings["qc"] - settings["auto_cutoff"][1]))].mean()
    primary = normalisator(settings["nom_scan_numbers"], settings["qc"], settings["footprint_correct"], settings["beam_width"], settings["sample_length"], settings["auto_cutoff_nom"])
    print(primary_ref/primary)
else:
    primary = settings["primary_intensity"]
    
# == check if path exist or create path
if settings["save_plot"] or settings["save_results"]:
    file_path = "{out_data_directory}/reflectivity/".format(**settings)
    try:
        os.makedirs(file_path)
    except OSError:
        if not os.path.isdir(file_path):
            raise

# == save settings dict to file
    setting_save = True
    out_setting_filename = file_path + "/{out_experiment}_{out_typ_experiment}_settings.txt".format(**settings)
    if os.path.exists(out_setting_filename):
        setting_save = input("Settings .txt output file already exists. Overwrite? [y/n] ") == "y"
    if setting_save == True:
        f = open(out_setting_filename,"w")
        f.write( str(settings) )
        f.close()
  
# == save data to file
if settings["save_results"]:
    out_filename = file_path + "/{out_experiment}_{out_typ_experiment}.dat".format(**settings)
    df = pandas.DataFrame()
    df["//qz"] = qz
    df["intensity_normalized"] = intensity / primary
    df["e_intensity_normalized"] = e_intensity / primary
    if os.path.exists(out_filename):
        settings["save_results"] = input("Results .dat output file already exists. Overwrite? [y/n] ") == "y"
    if settings["save_results"]:
        df.to_csv(out_filename, sep="\t", index=False)

# Calculation of the normalisation correction
test_inten = intensity[(qz > settings["auto_cutoff"][0]) & (qz<(settings["qc"] - settings["auto_cutoff"][1]))].mean()

inten_norm = intensity/primary
inten_norm_prim = inten_norm[(qz > settings["auto_cutoff"][0]) & (qz<(settings["qc"] - settings["auto_cutoff"][1]))].mean()


# == plot curve
rcParams["figure.figsize"] = 8, 5
rcParams["font.size"] = 16
#rcParams["text.usetex"] = False
#rcParams["text.latex.preamble"] = r"\usepackage{sfmath}"
fig = plt.figure()
fig.patch.set_color("white")
ax = fig.gca()
ax.set_yscale('log', nonposy='clip')
ax.errorbar(qz, intensity/primary, yerr=e_intensity/primary, ls='none',
            marker='o', mec='#cc0000', mfc='white', color='#ee0000',
            mew=1.2)
ax.errorbar(qz, fresnel(settings["qc"], qz, settings["roughness"]), ls='--', c='#424242')
ax.set_xlabel('q$_z$')
ax.set_ylabel('R')
ax.xaxis.set_minor_locator(MultipleLocator(0.05))
if settings["plot_title"] is not None:
    ax.set_title(settings["plot_title"])
plt.subplots_adjust(bottom=0.12)
if settings["show_plot"]:
    plt.show()
if settings["save_plot"]:
    out_plotname = file_path + "/{out_experiment}_{out_typ_experiment}.tiff".format(**settings)
    if os.path.exists(out_plotname):
        settings["save_plot"] = input("plot output file already exists. Overwrite? [y/n] ") == "y"
    if settings["save_plot"]:
        plt.savefig(out_plotname, dpi=300)
        
fig2 = plt.figure()
fig2.patch.set_color("white")
ax2 = fig2.gca()
plt.xlim([0.01,0.04])
plt.ylim([0.05,1.4])
#ax2.set_yscale('log', nonposy='clip')
ax2.errorbar(qz, intensity/primary, yerr=e_intensity/primary, ls='none',
            marker='o', mec='#cc0000', mfc='white', color='#ee0000',
            mew=1.2)
ax2.errorbar(qz, fresnel(settings["qc"], qz, settings["roughness"]), ls='--', c='#424242')
ax2.set_xlabel('q$_z$')
ax2.set_ylabel('R')
ax2.xaxis.set_minor_locator(MultipleLocator(0.05))

plt.show()