import copy
import os
import sys
from collections import namedtuple as natu

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio
import skimage.morphology
import tifffile as tif
from IPython import embed

sys.path.append("/home/jung/psd95/src")
from file_io import load_synfile
from preproc import get_hq_data,split_unsure_synapses


def extract(
    tifpath,
    synpath,
    bleedthrough,
    day=None,
    color="green",
    bg_sub=True,
    search_depth=5,
    integrate_over=5,
    hq_only=False
):
    # print("Extracting weights from {}".format(tifpath))
    frames = tif.imread(tifpath).astype("uint16")

    green_frames = frames[::2]
    red_frames = frames[1::2]

    noise_thresh = np.percentile(red_frames[:], 75, interpolation="midpoint")
    red_2 = red_frames[red_frames < noise_thresh]
    mean_red_2 = np.mean(red_2)
    std_red_2 = np.std(red_2)
    mask_thresh = mean_red_2 + 3 * std_red_2
    mask_img = red_frames > mask_thresh

    se2 = skimage.morphology.disk(2)
    se3 = skimage.morphology.disk(3)

    nu_img = np.zeros(mask_img.shape)

    for i, frame in enumerate(mask_img):
        closed = skimage.morphology.binary_closing(frame, se2)
        nu_img[i, :, :] = skimage.morphology.binary_opening(closed, se3)

    red_intensity_mask = nu_img
    red_hist = np.histogram(
        np.ravel(red_frames), np.arange(0, np.max(np.ravel(red_frames)))
    )
    most_abundant_value_red = np.argmax(red_hist[0], axis=0).astype("uint16")
    most_abundant_value_red = int(most_abundant_value_red)

    bleedthrough = np.multiply(bleedthrough, 0.01).astype("double")
    red_frames = np.array(red_frames).astype("uint16")
    green_frames = np.array(green_frames).astype("uint16")
    temp = np.zeros(red_frames.shape)
    red_frames = red_frames.astype("int16")
    most_abundant_value_red = np.array(most_abundant_value_red).astype("int16")

    temp = red_frames - most_abundant_value_red
    subtractionImg = np.multiply(temp, bleedthrough)
    subtractionImg[red_intensity_mask == 0] = 0
    subtractionImg = np.multiply(subtractionImg, red_intensity_mask)
    green_frames = green_frames - subtractionImg

    num_frames = green_frames.shape[0]

    syn_data = load_synfile(synpath)

    if hq_only:
        syn_data = get_hq_data(syn_data)

    num_syn = syn_data.synapseMatrix.shape[0]

    if day is None:
        bg_xc = int(syn_data.bgFluorData.xc - 1)
        bg_yc = int(syn_data.bgFluorData.yc - 1)
        bg_xr = int(syn_data.bgFluorData.xr)
        bg_yr = int(syn_data.bgFluorData.yr)
    else:
        bg_xc = int(syn_data.bgFluorData[day].xc - 1)
        bg_yc = int(syn_data.bgFluorData[day].yc - 1)
        bg_xr = int(syn_data.bgFluorData[day].xr)
        bg_yr = int(syn_data.bgFluorData[day].yr)

    bg_radii = (bg_xr, bg_yr)
    bg_centers = (bg_xc, bg_yc)

    g_bg = crop_ellipse(green_frames, bg_centers, bg_radii)
    r_bg = crop_ellipse(red_frames, bg_centers, bg_radii)

    bgmean = np.mean(g_bg, axis=1)
    rbgmean = np.mean(r_bg,axis=1)
    
    weights = []
    for syn in np.arange(num_syn):
        if day is None:
            is_syn = syn_data.synapseMatrix[syn] != 0
            is_zero = syn_data.synapseMatrix[syn] == 0

        else:
            is_syn = syn_data.synapseMatrix[syn, day] != 0
            is_zero = syn_data.synapseMatrix[syn, day] == 0

        if is_syn:
            if day is None:
                syn_center = [
                    int(syn_data.fluorData[syn].synX - 1),
                    int(syn_data.fluorData[syn].synY - 1),
                ]
                syn_radius = [
                    int(syn_data.fluorData[syn].xr),
                    int(syn_data.fluorData[syn].yr),
                ]

                guess_z = int(syn_data.synapseZ[syn] - 1)

            else:
                syn_center = [
                    int(syn_data.fluorData[syn][day].synX - 1),
                    int(syn_data.fluorData[syn][day].synY - 1),
                ]
                syn_radius = [
                    int(syn_data.fluorData[syn][day].xr),
                    int(syn_data.fluorData[syn][day].yr),
                ]
                guess_z = int(syn_data.synapseZ[syn,day] - 1)

            syn_green = crop_ellipse(green_frames, syn_center, syn_radius)
            syn_red = crop_ellipse(red_frames, syn_center, syn_radius)

            green_int = np.sum(syn_green, axis=1)
            red_int = np.sum(syn_red, axis=1)

            syn_greenbg = []
            syn_redbg = []

            for plane in range(num_frames):
                temp = syn_green[plane]
                syn_greenbg.append(temp - bgmean[plane])

                temp = syn_red[plane]
                syn_redbg.append(temp - rbgmean[plane])

            green_bgint = np.sum(np.array(syn_greenbg), axis=1)
            red_bgint = np.sum(np.array(syn_redbg), axis=1)

            if color == "green":
                if bg_sub:
                    found_z = search_frames(green_bgint, guess_z, search_depth)
                    weight = integrate_frames(green_bgint, found_z, integrate_over)
                else:
                    found_z = search_frames(green_int, guess_z, search_depth)
                    weight = integrate_frames(green_int, found_z, integrate_over)
            elif color == "red":
                if bg_sub:
                    found_z = search_frames(red_bgint, guess_z, search_depth)
                    weight = integrate_frames(red_bgint, found_z, integrate_over)
                else:
                    found_z = search_frames(red_int, guess_z, search_depth)
                    weight = integrate_frames(red_int, found_z, integrate_over)

        elif not is_syn:
            weight = float('nan')

        weights = np.append(weights, weight)
    return weights


def integrate_frames(vect, z, integrate_over):
    """integrate_frames.

    Parameters
    ----------
    vect :
        vect
    z :
        z
    integrate_over :
        integrate_over"""
    num_flanking = int((integrate_over - 1) / 2)
    assert len(vect) > num_flanking * 2
    start = z - num_flanking
    end = z + num_flanking + 1

    if start < 0:
        start = 0
        end = 0 + num_flanking * 2 + 1
        frames = vect[start:end]

    elif end >= len(vect):
        start = len(vect) - integrate_over
        frames = vect[start:]

    else:
        frames = vect[start:end]

    assert len(frames) == num_flanking * 2 + 1

    return np.sum(frames)


def search_frames(vect, guess, search_depth, plot=True):
    """search_frames.

    Parameters
    ----------
    vect :
        vect
    guess :
        guess
    search_depth :
        search_depth
    plot :
        plot
    """
    start = int(guess - (search_depth - 1) / 2)
    end = int(guess + (search_depth - 1) / 2) + 1
    num_frames = len(vect)

    if start < 0:
        start = 0
        end = search_depth - 1
        local = vect[start:end]

    elif end > len(vect):

        start = num_frames - search_depth
        local = vect[start:]

    else:
        local = vect[start:end]
    local_max = np.where(local == np.max(local))[0][0]
    max_frame = int(start + local_max)
    return max_frame


def crop_ellipse(raw, c_co, r_co):
    """crop_ellipse.

    Parameters
    ----------
    raw :
        raw
    c_co :
        c_co
    r_co :
        r_co
    """
    xc, yc = c_co
    xr, yr = r_co

    source = copy.deepcopy(raw)

    mask = np.zeros(source.shape[1:])
    mask = cv2.ellipse(
        mask,
        center=(int(xc), int(yc)),
        axes=(int(xr), int(yr)),
        angle=0,
        startAngle=0,
        endAngle=360,
        color=255,
        thickness=-1,
    ).astype("?")

    idxs = np.where(mask == True)
    pixels = source[:, idxs[0], idxs[1]]

    outside = np.where(mask == False)
    source[:, outside[0], outside[1]] = 0
    roi_img = source

    # print(c_co,r_co,pixels.shape)

    return pixels


def get_syn_img(raw, c_co, r_co):
    """crop_ellipse.

    Parameters
    ----------
    raw :
        raw
    c_co :
        c_co
    r_co :
        r_co
    """
    xc, yc = c_co
    xr, yr = r_co

    source = copy.deepcopy(raw)

    mask = np.zeros(source.shape[1:])
    mask = cv2.ellipse(
        mask,
        center=(int(xc), int(yc)),
        axes=(int(xr), int(yr)),
        angle=0,
        startAngle=0,
        endAngle=360,
        color=255,
        thickness=-1,
    ).astype("?")

    idxs = np.where(mask == True)
    pixels = source[:, idxs[0], idxs[1]]

    outside = np.where(mask == False)
    source[:, outside[0], outside[1]] = 0
    roi_img = source

    return roi_img

def normalize(syn_by_day, method="quantile", q=[40, 60]):
    """normalize.

    Parameters
    ----------
    syn_by_day :
        syn_by_day
    """
    if method == "quantile":
        # print('Normalizing via quantile at {}'.format(q))
        qr = np.nanpercentile(syn_by_day, q=q, interpolation="nearest", axis=0).T
        syn_sorted = np.sort(syn_by_day, axis=0)

        avg = []

        for i, q in enumerate(qr):
            syn_day = syn_sorted[:, i]
            gt = syn_day[syn_day >= q[0]]
            lt = gt[gt <= q[1]]
            avg.append(np.nanmean(lt))

        avg = np.array(avg)

    elif method == "median":
        # print('Normalizing via median')
        avg = np.nanmedian(syn_by_day, axis=0)
    elif method == "mean":
        # print('Normalizing via mean')
        avg = np.nanmean(syn_by_day, axis=0)

    norm = syn_by_day / avg

    return norm


def construct_weight_matrix(synpath, tifdir, norm=True, channel='green',search=7,integrate=5,bleedthrough=2,hq_only=True,split_unsures=True):
    data = load_synfile(synpath)
    if hq_only:
        data = get_hq_data(data)

    num_syn, num_days = data.synapseMatrix.shape

    dend_id = os.path.split(synpath)[1].split(".")[0].split("_")[2][3:]
    group_path = tifdir + "{}/{}/{}.grp".format(dend_id[:2], dend_id, dend_id)
    group = spio.loadmat(
        group_path, struct_as_record=False, squeeze_me=True, chars_as_strings=True
    )["groupFiles"]
    # print(group)
    weights = np.zeros([num_syn, num_days])

    for day in np.arange(num_days):
        fname = group[day].fname
        impath = os.path.join(tifdir, dend_id[:2], dend_id, fname)
        weight_on_day = extract(impath,synpath,bleedthrough=2,day=day,search_depth=search,integrate_over=integrate,hq_only=hq_only,color=channel)
        weights[:,day] = weight_on_day
    if norm:
        weights = normalize(weights)

    if split_unsures:
        weights = split_unsure_synapses(data.synapseMatrix,weights)

    return weights
