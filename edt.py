import sys
import scipy
import cv2 as cv
import numpy as np
import pandas as pd
from scipy import ndimage
from matplotlib import pyplot as plt
from PIL import Image
import skimage as ski
import pytesseract
from edt_utils import (
    is_nan, py_blockproc, display_segments, detect_ref_pulse,
    print_line_dict, segment_to_df, remove_text,
    process_line, get_values_from_img, measure_extract_pulse,
    plot_ecg, extract_image
)
from ss import pattern_match
from scipy.signal import find_peaks
import operator


# =============================================================================
# HELPER — plot inside GUI or in a standalone window
# =============================================================================

def _show(fig, plot_callback):
    """
    If a plot_callback is provided (GUI mode), send the figure to the GUI.
    Otherwise call plt.show() as usual (standalone mode).
    In both cases the figure is properly closed afterwards to free memory.
    """
    if plot_callback is not None:
        plot_callback(fig)
        plt.close(fig)
    else:
        plt.show()


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def ecg_to_csv(image_name, template_name, csv_name, config_dict):
    """
    Convert an ECG image to a CSV file with the digitised lead signals.

    Parameters
    ----------
    image_name    : path to the ECG image
    template_name : path to the calibration-pulse template image
    csv_name      : output CSV path
    config_dict   : dictionary with all processing parameters (see below)

    Key config_dict entries
    -----------------------
    layout          : tuple (rows, cols) — e.g. (3, 4) or (12, 1)
    pulse           : 0 | -1 | int | list[int]
    rhythm          : int  (row index of rhythm lead, 0 = none)
    verbose         : int  (0 = silent, >0 = progressively more info)
    show_images     : bool (True = emit debug plots)
    plot_callback   : callable(fig) | None
                      When not None, every debug figure is passed to this
                      function instead of calling plt.show().  The GUI
                      supplies self.add_debug_figure here; standalone runs
                      leave it as None so plt.show() is used normally.
    strategy        : 'none' | 'filter' | 'color'
    thres_value     : int   binarisation threshold (0-255)
    lower / upper   : HSV tuples for the 'color' strategy
    kSized2d/1d     : kernel sizes for the 'filter' strategy
    perc_space_leads: float — minimum fractional distance between lead peaks
    dilation        : int   — morphological dilation iterations
    perc_max_dist   : float — fraction of peak distance used to crop each row
    mmpsec          : float — paper speed in mm/s  (default 25)
    mmpmv           : float — amplitude scale in mm/mV (default 10)
    pulse_width_mm  : float — calibration pulse width in mm  (default 5)
    pulse_height_mm : float — calibration pulse height in mm (default 10)
    sample_frequency: int   — target sampling rate after interpolation
    time_lead       : float — duration of each lead segment in seconds
    num_sampling_points: int
    location        : 'right' | 'left'
    """

    # ------------------------------------------------------------------
    # Unpack config
    # ------------------------------------------------------------------
    show_images     = config_dict.get('show_images', False)
    plot_callback   = config_dict.get('plot_callback', None)   # ← NEW

    layout          = config_dict['layout']
    pulse           = config_dict['pulse']
    rhythm          = config_dict['rhythm']
    verbose         = config_dict['verbose']
    strategy        = config_dict['strategy']
    thres_value     = config_dict['thres_value']
    lower           = config_dict['lower']
    upper           = config_dict['upper']
    kSize2d         = config_dict['kSized2d']
    kSize1d         = config_dict['kSized1d']
    perc_space_leads= config_dict['perc_space_leads']
    dilation        = config_dict['dilation']
    perc_max_dist   = config_dict['perc_max_dist']
    pulse_per_sec   = config_dict['pulse_per_sec']
    pulse_per_mv    = config_dict['pulse_per_mv']
    num_sampling_points = config_dict['num_sampling_points']

    # ------------------------------------------------------------------
    # Lead name table — depends on layout
    # ------------------------------------------------------------------
    if layout[1] == 4 and layout[0] == 3:
        lt_leads = [
            ['I',   'aVR', 'V1', 'V4'],
            ['II',  'aVL', 'V2', 'V5'],
            ['III', 'aVF', 'V3', 'V6'],
            ['II']
        ]
    elif layout[1] == 2:
        raise NotImplementedError('2-column layout not yet implemented')
    elif layout[1] == 1:
        lt_leads = [
            ['I'], ['II'], ['III'],
            ['aVR'], ['aVL'], ['aVF'],
            ['V1'], ['V2'], ['V3'],
            ['V4'], ['V5'], ['V6']
        ]
    else:
        raise ValueError('layout columns must be 4, 2 or 1')

    # ------------------------------------------------------------------
    # Pulse detection setup
    # ------------------------------------------------------------------
    if pulse == 0:
        print("INFO: No pulse to be detected")
    elif pulse == -1:
        print("INFO: pulse to be detected in all lines")
    elif isinstance(pulse, list):
        print("INFO: pulse on lines: {}.".format(pulse))
        for p in pulse:
            lt_leads[p].append('Pulse')
    elif isinstance(pulse, int):
        print("INFO: pulse on line: {}.".format(pulse))
        lt_leads[pulse].append('Pulse')
    else:
        raise ValueError('pulse should be 0, an int or a list')

    # ------------------------------------------------------------------
    # Rhythm
    # ------------------------------------------------------------------
    if rhythm == 0:
        print("INFO: No rhythm lead")
    else:
        print("INFO: rhythm on line: {}.".format(rhythm))

    # ------------------------------------------------------------------
    # Load image
    # ------------------------------------------------------------------
    image = cv.imread(image_name)
    if image is None:
        raise ValueError(f"Cannot open image: {image_name}")

    if show_images and verbose > 1:
        fig = plt.figure()
        plt.imshow(image)
        plt.title("Original Image")
        _show(fig, plot_callback)
        print("INFO: Image Shape {}.".format(image.shape))

    # ------------------------------------------------------------------
    # Pre-processing strategy
    # ------------------------------------------------------------------
    if strategy == 'color':
        img_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        mask = cv.inRange(img_hsv, lower, upper)
        result = img_hsv.copy()
        result[mask != 255] = (255, 255, 255)
        image = cv.cvtColor(result, cv.COLOR_HSV2BGR)
        image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        ret, th1 = cv.threshold(image_gray, thres_value, 255, cv.THRESH_BINARY)

    elif strategy == 'filter':
        image_gray = extract_image(image, kSize2d, kSize1d)
        ret, th1 = cv.threshold(image_gray, thres_value, 255, cv.THRESH_BINARY)

    else:  # 'none'
        image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        ret, th1 = cv.threshold(image_gray, thres_value, 255, cv.THRESH_BINARY)

    # ------------------------------------------------------------------
    # Debug plots — gray scale
    # ------------------------------------------------------------------
    if show_images and verbose > 0:
        fig, ax = plt.subplots()
        ax.imshow(image_gray, cmap="gray")
        ax.set_title("Gray Scale")
        _show(fig, plot_callback)
        print("INFO: gray scale image Shape {}.".format(image_gray.shape))

    # ------------------------------------------------------------------
    # Debug plots — binary image
    # ------------------------------------------------------------------
    if show_images and verbose > 2:
        fig, ax = plt.subplots()
        ax.imshow(th1, cmap="gray")
        ax.set_title("Binary Image")
        _show(fig, plot_callback)
        print("INFO: Binary image Shape {}.".format(th1.shape))

    # ------------------------------------------------------------------
    # Foreground (inverted + optional dilation)
    # ------------------------------------------------------------------
    if dilation != 0:
        foreground = cv.morphologyEx(
            255 - th1, cv.MORPH_DILATE, np.ones((3, 3)), iterations=dilation
        )
    else:
        foreground = 255 - th1

    # ------------------------------------------------------------------
    # Load calibration-pulse template
    # ------------------------------------------------------------------
    template = cv.imread(template_name, cv.IMREAD_GRAYSCALE)
    if template is None:
        raise ValueError(f"Cannot open template: {template_name}")

    _, new_template = cv.threshold(template, 127, 255, cv.THRESH_OTSU)
    new_template = (new_template != 255) * np.uint8(255)

    if show_images and verbose > 2:
        fig, ax = plt.subplots()
        ax.imshow(new_template, cmap="gray")
        ax.set_title("Template")
        _show(fig, plot_callback)

    # ------------------------------------------------------------------
    # Detect lead rows using vertical std-dev profile + peak finding
    # ------------------------------------------------------------------
    temp = py_blockproc(foreground, (1, foreground.shape[1]), func=0)
    median_temp = np.median(temp.flatten())
    peak_indices, peak_dict = find_peaks(
        temp.flatten(),
        height=median_temp,
        distance=round(temp.flatten().size * perc_space_leads, 0)
    )
    peak_heights = peak_dict['peak_heights']
    highest_peak_index = peak_indices[np.argsort(peak_heights)]

    if show_images and verbose > 0:
        fig, ax = plt.subplots()
        ax.plot(temp.flatten(), label="Std-dev profile")
        ax.plot(
            highest_peak_index[-(layout[0] + 1):],
            temp[highest_peak_index[-(layout[0] + 1):]],
            "x", label="Selected peaks"
        )
        ax.plot(
            median_temp * np.ones_like(temp), "--",
            color="gray", label="Median threshold"
        )
        ax.set_title("Peak Detection")
        ax.legend()
        _show(fig, plot_callback)

    # ------------------------------------------------------------------
    # Calculate row crop boundaries
    # ------------------------------------------------------------------
    ordered_hp_index = sorted(highest_peak_index[-(layout[0] + 1):])
    peak_dist = [
        np.abs(t - s)
        for s, t in zip(ordered_hp_index, ordered_hp_index[1:])
    ]
    max_dist = int(np.round(max(peak_dist) * perc_max_dist, 0))

    slices_x = [
        (max(0, s - max_dist), min(foreground.shape[0], s + max_dist), None)
        for s in ordered_hp_index
    ]

    if verbose > 0:
        print("INFO: slices: {}".format(slices_x))

    # ------------------------------------------------------------------
    # Process each lead row
    # ------------------------------------------------------------------
    proc_line_list = []

    # wt / ht are computed from the template and used as fallback values
    _, _, xt, yt = get_values_from_img(new_template)
    wt, ht = measure_extract_pulse(xt, yt, verbose=0)

    for i, slx in enumerate(slices_x):

        line = foreground[slice(*slx), slice(0, foreground.shape[1], None)]
        offset = slx

        if show_images and verbose > 0:
            fig, ax = plt.subplots()
            ax.imshow(line, cmap="gray")
            ax.set_title(f"Line {i}")
            _show(fig, plot_callback)

        structure = np.array(
            [[1, 1, 1],
             [1, 1, 1],
             [1, 1, 1]], np.uint8
        )
        labeled_line, nb = ndimage.label(line, structure=structure)

        if show_images and verbose > 0:
            print("INFO: Number of segments {} on line {}.".format(nb, i))
            display_segments(
                'Labeled line ' + str(i), labeled_line,
                plot_callback=plot_callback      # ← pass callback
            )

        # --------------------------------------------------------------
        # Pulse detection on this row
        # --------------------------------------------------------------
        if (pulse == -1) or (i in pulse):

            line_signal = (labeled_line != 0) * np.uint8(255)
            line_copy = line_signal.copy()

            template_width, template_height = template.shape

            config_dict['hpulse'] = ht   # default from template measurement
            config_dict['wpulse'] = wt

            detected, location, similarity_value, x, y, wpulse, hpulse = \
                detect_ref_pulse(line_copy, new_template)

            print("INFO: line {}: best similarity value = {} in {}".format(
                i, similarity_value, y))

            if detected:
                if location == 'right':
                    sliced_labeled_line = labeled_line[:, 0:y].copy()
                elif location == 'left':
                    sliced_labeled_line = labeled_line[:, y + int(wpulse):].copy()
                else:
                    sliced_labeled_line = labeled_line.copy()
            else:
                if is_nan(wpulse):
                    wpulse = wt
                    hpulse = ht
                sliced_labeled_line = labeled_line.copy()

            if verbose > 0:
                if detected:
                    print('INFO: pulse detected by template in line {} in {}'.format(i, y))
                    if show_images:
                        fig, ax = plt.subplots()
                        ax.imshow(
                            line_copy[
                                x:x + int(template_width) + 1,
                                y:y + int(template_height) + 1
                            ],
                            cmap="gray"
                        )
                        ax.set_title(f"Pulse Detection Line {i}")
                        _show(fig, plot_callback)
                else:
                    print('INFO: pulse NOT detected by template in line {}'.format(i))

        else:
            print("INFO: line {} has no pulse to detect".format(i))
            wpulse = wt
            hpulse = ht
            sliced_labeled_line = labeled_line.copy()

        config_dict['wpulse'] = wpulse
        config_dict['hpulse'] = hpulse

        # --------------------------------------------------------------
        # Process lead row → extract signal segments
        # --------------------------------------------------------------
        line_dict = process_line(
            i, sliced_labeled_line, offset, lt_leads[i],
            config_dict, config_dict['verbose'],
            plot_callback=plot_callback       # ← pass callback
        )
        proc_line_list.append(line_dict)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    for i, line in enumerate(proc_line_list):
        print("INFO: processing line {}".format(i))
        if verbose > 0:
            print_line_dict(line, plot_callback=plot_callback)
        else:
            for key, value in line.items():
                if key != 'curves':
                    print(f"{key}: {value}")

    # ------------------------------------------------------------------
    # Remove rhythm row from list before building the DataFrame
    # ------------------------------------------------------------------
    if config_dict['rhythm'] != 0:
        proc_line_list.pop(rhythm - 1)

    # ------------------------------------------------------------------
    # Build DataFrame and save CSV
    # ------------------------------------------------------------------
    ecg_df = segment_to_df(proc_line_list, pulse_per_sec, pulse_per_mv, num_sampling_points)
    ecg_df.to_csv(csv_name)

    return {
        "df": ecg_df,
        "foreground": foreground,
        "binary": th1,
        "image_gray": image_gray
    }


# =============================================================================
# DEFAULT CONFIG — used when running edt.py directly (standalone mode)
# =============================================================================

filename         = 'ecg2'
image_name       = filename + '.png'
template_name    = 'pul.png'
csv_name         = filename + '.csv'

layout           = (3, 4)
pulse            = [0, 1, 2]
rhythm           = 4
verbose          = 0
mmpsec           = 25
mmpmv            = 10
pulse_width_mm   = 5
pulse_height_mm  = 10
pulse_per_sec    = pulse_width_mm / mmpsec
pulse_per_mv     = pulse_height_mm / mmpmv
sample_frequency = 500
time_lead        = 2.5
num_sampling_points = int(time_lead * sample_frequency)
location         = 'right'
strategy         = 'none'
lower            = (0,   0,   0)
upper            = (179, 255, 220)
thres_value      = 127
kSize2d          = 3
kSize1d          = 3
perc_space_leads = 0.2
dilation         = 10
perc_max_dist    = 0.7

config_dict = {
    'pulse':              pulse,
    'rhythm':             rhythm,
    'verbose':            verbose,
    'show_images':        False,
    'plot_callback':      None,        # None → standalone plt.show() behaviour
    'mmpsec':             mmpsec,
    'mmpmv':              mmpmv,
    'pulse_width_mm':     pulse_width_mm,
    'pulse_height_mm':    pulse_height_mm,
    'pulse_per_sec':      pulse_per_sec,
    'pulse_per_mv':       pulse_per_mv,
    'sample_frequency':   sample_frequency,
    'time_lead':          time_lead,
    'location':           location,
    'layout':             layout,
    'num_sampling_points':num_sampling_points,
    'strategy':           strategy,
    'lower':              lower,
    'upper':              upper,
    'thres_value':        thres_value,
    'kSized2d':           kSize2d,
    'kSized1d':           kSize1d,
    'perc_space_leads':   perc_space_leads,
    'dilation':           dilation,
    'perc_max_dist':      perc_max_dist,
}


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    result = ecg_to_csv(image_name, template_name, csv_name, config_dict)

    df = result["df"]

    plot_ecg(
        df, df.columns, csv_name,
        n_rows=layout[0],
        n_columns=layout[1],
        fs=500,
        figure_size=(20, 12)
    )

    print("THE END")
