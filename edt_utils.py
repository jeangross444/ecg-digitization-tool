from matplotlib import pyplot as plt
from itertools import groupby
from scipy import interpolate
from scipy import ndimage
from PIL import Image
import skimage as ski
import pandas as pd
import numpy as np
import cv2 as cv
import scipy
import sys
import pytesseract
import pprint
import math
import ss


# =============================================================================
# INTERNAL HELPER
# =============================================================================

def _show(fig, plot_callback):
    """
    Route a matplotlib figure to the right destination.

    - GUI mode  (plot_callback is not None): send figure to the callback
      (which renders it inside the debug tab) and close it to free memory.
    - Standalone mode (plot_callback is None): call plt.show() as usual.
    """
    if plot_callback is not None:
        plot_callback(fig)
        plt.close(fig)
    else:
        plt.show()


# =============================================================================
# IMAGE PROCESSING
# =============================================================================

def laplacian_filter(img, kSize=3, gSize=3, alpha=1.0):
    """Sharpen an image using a Laplacian high-pass filter."""
    input_is_bgr = len(img.shape) == 3

    if input_is_bgr:
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        gray_img = img

    # Gaussian blurring / low-pass filter.
    gauss = cv.GaussianBlur(gray_img, (gSize, gSize), 0.0)

    # Edge detection / high-pass filter.
    lpl = cv.Laplacian(gauss, cv.CV_32F, ksize=kSize)
    if input_is_bgr:
        lpl = cv.cvtColor(lpl, cv.COLOR_GRAY2BGR)

    # Image sharpening.
    filtered_img = img.astype("float32") - alpha * lpl

    return np.clip(filtered_img, 0.0, 255.0).astype("uint8")


def extract_sequence(cropped_signal, kSize2d=3, kSize1d=3):
    """
    Extract the ECG signal skeleton as a 1-D sequence of row indices.
    Based on https://github.com/alphanumericslab/ecg-image-kit.
    """
    if len(cropped_signal == 3):
        gray_signal = cv.cvtColor(cropped_signal, cv.COLOR_BGR2GRAY)
    else:
        gray_signal = cropped_signal

    lpl_signal = laplacian_filter(gray_signal, 3, 7, 0.5)

    argmin_seq    = lpl_signal.argmin(axis=0)
    box1d_seq     = cv.filter2D(lpl_signal, cv.CV_32F,
                                np.ones((1, kSize1d), np.float32) / kSize1d
                                ).argmin(axis=0)
    box2d_seq     = cv.filter2D(lpl_signal, cv.CV_32F,
                                np.ones((kSize2d, kSize2d), np.float32) / (kSize2d * kSize2d)
                                ).argmin(axis=0)

    h0 = np.array([[1., 0., 1.],
                   [1., 1., 1.],
                   [1., 0., 1.]], np.float32) / 7.
    lr_neigh_seq  = cv.filter2D(lpl_signal, cv.CV_32F, h0).argmin(axis=0)

    h1 = np.array([1., 1., 1.], np.float32) / 3.
    z1 = cv.filter2D(lpl_signal, cv.CV_32F, h1).argmin(axis=0)

    h2 = np.array([[1., 0., 0.],
                   [0., 1., 0.],
                   [0., 0., 1.]], np.float32) / 3.
    z2 = cv.filter2D(lpl_signal, cv.CV_32F, h2).argmin(axis=0)

    h3 = np.array([[0., 0., 1.],
                   [0., 1., 0.],
                   [1., 0., 0.]], np.float32) / 3.
    z3 = cv.filter2D(lpl_signal, cv.CV_32F, h3).argmin(axis=0)

    all_neigh_seq = np.maximum(np.maximum(z1, z2), z3)

    return np.median([argmin_seq, box1d_seq, box2d_seq,
                      lr_neigh_seq, all_neigh_seq], axis=0)


def extract_image(cropped_img, kSize2d=3, kSize1d=3):
    """
    Produce a filtered/sharpened greyscale image suitable for binarisation.
    Based on https://github.com/alphanumericslab/ecg-image-kit.
    """
    if len(cropped_img == 3):
        gray_img = cv.cvtColor(cropped_img, cv.COLOR_BGR2GRAY)
    else:
        gray_img = cropped_img

    lpl_img      = laplacian_filter(gray_img, 3, 7, 0.5)
    box1d_img    = cv.filter2D(lpl_img, cv.CV_32F,
                               np.ones((1, kSize1d), np.float32) / kSize1d)
    box2d_img    = cv.filter2D(lpl_img, cv.CV_32F,
                               np.ones((kSize2d, kSize2d), np.float32) / (kSize2d * kSize2d))

    h0 = np.array([[1., 0., 1.],
                   [1., 1., 1.],
                   [1., 0., 1.]], np.float32) / 7.
    lr_neigh_img = cv.filter2D(lpl_img, cv.CV_32F, h0)

    h1 = np.array([1., 1., 1.], np.float32) / 3.
    z1 = cv.filter2D(lpl_img, cv.CV_32F, h1)

    h2 = np.array([[1., 0., 0.],
                   [0., 1., 0.],
                   [0., 0., 1.]], np.float32) / 3.
    z2 = cv.filter2D(lpl_img, cv.CV_32F, h2)

    h3 = np.array([[0., 0., 1.],
                   [0., 1., 0.],
                   [1., 0., 0.]], np.float32) / 3.
    z3 = cv.filter2D(lpl_img, cv.CV_32F, h3)

    all_neigh_img = np.maximum(np.maximum(z1, z2), z3)

    output_img = np.median([lpl_img, box1d_img, box2d_img,
                            lr_neigh_img, all_neigh_img], axis=0)
    output_img = cv.normalize(output_img, None, 255, 0, cv.NORM_MINMAX, cv.CV_8U)
    return output_img


# =============================================================================
# UTILITIES
# =============================================================================

def is_nan(value):
    try:
        return math.isnan(float(value))
    except ValueError:
        return False


def get_rectangular_contours(contours):
    """Return only the contours that approximate to a rectangle (4 vertices)."""
    res = []
    for contour in contours:
        hull  = cv.convexHull(contour)
        peri  = cv.arcLength(hull, closed=True)
        approx = cv.approxPolyDP(hull, 0.04 * peri, closed=True)
        if len(approx) == 4:
            res.append(approx)
    return res


def py_blockproc(A, blockdims, func=0):
    """Compute per-block standard deviation (mimics MATLAB's blockproc)."""
    vr = A.shape[0] // blockdims[0]
    hr = A.shape[1] // blockdims[1]
    B  = np.zeros((vr, hr))

    for i in range(vr):
        for j in range(hr):
            B[i, j] = np.std(
                A[i * blockdims[0]:(i + 1) * blockdims[0],
                  j * blockdims[1]:(j + 1) * blockdims[1]]
            )
    return B


# =============================================================================
# DISPLAY
# =============================================================================

def display_segments(name, item, axis='off', plot_callback=None):
    """
    Display a labelled/segmented image.

    Parameters
    ----------
    name          : figure title
    item          : 2-D array to display
    axis          : 'off' | 'on'  (matplotlib axis visibility)
    plot_callback : callable(fig) | None
                    When provided the figure is sent to the GUI debug tab
                    instead of calling plt.show().
    """
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(item, cmap="magma")
    ax.set_title(name)
    ax.axis(axis)
    fig.subplots_adjust(wspace=0.05, left=0.01, bottom=0.01, right=0.99, top=0.9)
    _show(fig, plot_callback)


# =============================================================================
# SIGNAL EXTRACTION
# =============================================================================

def get_values_from_img(roi):
    """
    Extract the (x, y) skeleton of the ECG trace from a binary ROI image.

    Parameters
    ----------
    roi : 2-D binary array — white pixels are the signal

    Returns
    -------
    width, length, xs, ys
    """
    def find_nearest(array, value):
        array = np.asarray(array)
        return array[(np.abs(array - value)).argmin()]

    width, length = roi.shape[:2]
    xs, ys = [], []
    bool_roi   = roi != 0
    old_dy_dx  = 0.0

    for i, col in enumerate(bool_roi.T):
        if len(xs) != 0:
            label, num = ndimage.label(col, structure=np.ones((3,)))
            if num != 0:
                median_list   = []
                dy_dx_list    = []
                d2y_dx2_list  = []
                for n in range(1, num + 1):
                    pixel_loc    = width - np.where(label == n)[0]
                    median_pixel = find_nearest(pixel_loc, np.median(pixel_loc))
                    median_list.append(median_pixel)
                    dy_dx  = median_pixel - ys[-1]
                    d2y_dx2 = dy_dx - old_dy_dx
                    dy_dx_list.append(dy_dx)
                    d2y_dx2_list.append(d2y_dx2)
                tmp = np.argmin(np.abs(d2y_dx2_list))
                old_dy_dx = dy_dx_list[tmp]
                xs.append(i)
                ys.append(median_list[tmp])
        else:
            pixel_loc = width - np.where(col)[0]
            if pixel_loc.size > 0:
                xs.append(i)
                ys.append(np.median(pixel_loc))

    return width, length, xs, ys


def measure_extract_pulse(x, y, verbose=0):
    """Measure the width and height of a calibration pulse in pixel units."""
    min_pulse = np.min(y)
    max_pulse = np.max(y)
    height    = np.max(max_pulse - min_pulse)
    threshold = height / 2
    index     = np.where((y - min_pulse) >= threshold)[0]
    width     = x[index[-1]] - x[index[0]]
    if verbose > 0:
        print(f"pulse height: {height}")
        print(f"pulse width:  {width} time units")
    return width, height


def convert_to_secmv(xs, ys, wp, hp, ws, baseline, pulse_per_sec, pulse_per_mv):
    """
    Convert pixel coordinates to physical units (seconds and millivolts).

    Parameters
    ----------
    xs, ys        : signal coordinates in pixels
    wp            : pulse width  in pixels
    hp            : pulse height in pixels
    ws            : segment width in pixels  (unused — kept for API compat.)
    baseline      : row index of the signal baseline
    pulse_per_sec : seconds per mm of pulse width
    pulse_per_mv  : mV     per mm of pulse height
    """
    zero_line   = ws - baseline
    ymv         = (ys - zero_line) / (hp * pulse_per_mv)
    sec_per_pts = pulse_per_sec / wp
    xsec        = sec_per_pts * np.asarray(xs)
    return xsec, ymv


# =============================================================================
# PULSE DETECTION
# =============================================================================

def detect_ref_pulse(roi, template, location='right', threshold=0.6, verbose=2):
    """
    Locate the calibration pulse inside a lead row using template matching.

    Returns
    -------
    detected        : bool
    location        : 'right' | 'left'
    similarity_value: float — best match score
    x, y            : top-left corner of the best match (row, col)
    wpulse, hpulse  : measured pulse width and height in pixels
                      (np.nan when not detected)
    """
    if roi.shape[0] <= template.shape[0] or roi.shape[1] <= template.shape[1]:
        # Template is larger than the ROI — matching is impossible.
        empty = np.array([])
        loc   = (empty, empty)
        # Initialise these so the name is always bound before the return.
        x = y = 0
        similarity_value = 0.0
    else:
        method = cv.TM_CCORR_NORMED
        res    = cv.matchTemplate(roi, template, method)

        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left         = min_loc
            similarity_value = min_val
            print("INFO: min similarity value is {} in x={} y={}.".format(
                min_val, top_left[1], top_left[0]))
        else:
            top_left         = max_loc
            similarity_value = max_val
            print("INFO: max similarity value is {} in x={} y={}.".format(
                max_val, top_left[1], top_left[0]))

        x = top_left[1]
        y = top_left[0]

        loc = np.where(res >= threshold)

    if len(loc[0]) > 0:
        detected = True

        ppts        = np.array(list(map(list, zip(*loc[::-1]))))
        template_width, template_height = template.shape

        extracted_pulse = roi[x:x + template_width, y:y + template_height]
        _, _, xpulse, ypulse = get_values_from_img(extracted_pulse)
        wpulse, hpulse = measure_extract_pulse(xpulse, ypulse)
    else:
        detected = False
        wpulse   = np.nan
        hpulse   = np.nan

    return detected, location, similarity_value, x, y, wpulse, hpulse


# =============================================================================
# PRINT / DEBUG HELPERS
# =============================================================================

def print_segment_list(segment_list, plot_callback=None):
    """Print metadata and plot each segment curve.
    Plots are only emitted when plot_callback is provided (GUI mode)
    or when running standalone with plt.show() explicitly requested.
    When plot_callback is None and running from the GUI, no plot is shown.
    """
    for seg in segment_list:
        print("line number: {} - name: {} - segment length: {}".format(
            seg['line'], seg['name'], seg['lseg']))
        if plot_callback is not None:
            fig, ax = plt.subplots()
            ax.set_title(seg['name'])
            ax.plot(seg['xseg'], seg['yseg'])
            ax.grid(True)
            _show(fig, plot_callback)


def print_line_dict(line, plot_callback=None):
    """Pretty-print a line dictionary; delegate curve plots to print_segment_list."""
    for key, value in line.items():
        if key == 'curves':
            print_segment_list(value, plot_callback=plot_callback)
        else:
            print(f"{key}: {value}")


# =============================================================================
# INTERPOLATION & DATAFRAME CONSTRUCTION
# =============================================================================

def interpolate_segment(x, y, num):
    """Resample a segment to *num* evenly-spaced points using cubic splines."""
    x_interp = np.linspace(0.0, 1.0, len(x))
    f        = interpolate.CubicSpline(x_interp, y)
    x_new    = np.linspace(0.0, 1.0, int(num))
    y_new    = f(x_new)
    return x_new, y_new


def segment_to_df(line_list, pulse_per_sec, pulse_per_mv, num_pts):
    """
    Convert a list of processed lead rows into a tidy pandas DataFrame.

    Each column corresponds to one ECG lead, resampled to *num_pts* points.
    """
    df = pd.DataFrame()
    for line in line_list:
        for seg in line['curves']:
            xsec, ymv = convert_to_secmv(
                seg['xseg'], seg['yseg'],
                line['wpulse'], line['hpulse'],
                seg['wseg'], seg['baseline'],
                pulse_per_sec, pulse_per_mv
            )
            x_new, y_new  = interpolate_segment(xsec, ymv, num_pts)
            df[seg['name']] = y_new
    return df


# =============================================================================
# TEXT REMOVAL
# =============================================================================

def remove_text(image, confidence_threshold):
    """
    Use pytesseract to detect text regions and paint them black.

    Parameters
    ----------
    image                : BGR image (numpy array)
    confidence_threshold : float in [0, 1] — only detections above this are erased
    """
    image_copy = image.copy()
    results    = pytesseract.image_to_data(
        image_copy, config='--psm 11', output_type='dict'
    )
    for i in range(len(results["text"])):
        x, y, w, h = (results["left"][i], results["top"][i],
                      results["width"][i], results["height"][i])
        conf = int(results["conf"][i])
        if conf > 100 * confidence_threshold:
            print("INFO: word detected in the image")
            cv.rectangle(image_copy, (x, y), (x + w, y + h), (0, 0, 0), -1)
    return image_copy


# =============================================================================
# CORE LINE PROCESSOR
# =============================================================================

def _plot_seg(seg, title, verbose, plot_callback):
    """Helper: plot a segment array when verbose > 0."""
    if verbose > 0:
        fig, ax = plt.subplots()
        ax.imshow(seg)
        ax.set_title(title)
        _show(fig, plot_callback)


def process_line(line_number, labeled_line, offset, line_leads,
                 config_dict, verbose=0, plot_callback=None):
    """
    Extract individual lead segments from one labelled row of the ECG image.

    Parameters
    ----------
    line_number   : int   — 0-based row index
    labeled_line  : 2-D int array — output of scipy.ndimage.label
    offset        : tuple — (start_x, stop_x, None) used to record absolute position
    line_leads    : list[str] — expected lead names for this row
    config_dict   : dict  — processing parameters
    verbose       : int   — 0 = silent, >0 = progressively more output
    plot_callback : callable(fig) | None
                    Routing function for debug figures (see _show).

    Returns
    -------
    line_dict : dict with keys 'wpulse', 'hpulse', 'curves', 'offset_line'
    """
    line_dict = {
        'wpulse':      config_dict['wpulse'],
        'hpulse':      config_dict['hpulse'],
        'curves':      [],
        'offset_line': offset,
    }

    # ── Debug: show the full labelled row ────────────────────────────────────
    if verbose > 0:
        display_segments(
            "Labeled Line " + str(line_number),
            labeled_line,
            plot_callback=plot_callback
        )

    # ── Segment statistics ────────────────────────────────────────────────────
    u, c            = np.unique(labeled_line, return_counts=True)
    segment_labels  = np.argsort(-c[1:]) + 1   # sorted by size, descending
    max_label       = np.max(u)
    app_seg_size    = labeled_line.shape[1] // config_dict['layout'][1]

    if verbose > 1:
        print("INFO: unique label {}.".format(u))
        print("INFO: count {}.".format(c))
        print("INFO: segment labels {}.".format(segment_labels))

    segment_ratios = []
    min_roi_length = np.round(app_seg_size * 0.25, 0)

    # ── Iterate over segments sorted by size ─────────────────────────────────
    for l, label in enumerate(segment_labels):

        roi = (labeled_line == label)
        sl  = ndimage.find_objects(roi.astype(np.int32))
        if len(sl) == 0:
            continue

        roi        = roi[sl[0][0], sl[0][1]]
        roi_length = roi.shape[1]

        if roi_length < min_roi_length:
            continue   # too small — skip

        roi_copy = roi * np.uint8(255)
        ratio    = round(roi_length / app_seg_size, 0)
        segment_ratios.append(ratio)

        if verbose > 0:
            print("INFO: label = {}, length = {}, ratio = {}.".format(
                label, roi_copy.shape[1], ratio))

        # Skip rhythm row entirely
        if line_number + 1 == config_dict['rhythm']:
            continue

        # ── ratio == 4: four merged segments (not yet split — logged only) ───
        if ratio == 4.0:
            print("INFO: Four Segments {}.".format(ratio))
            print("INFO: Slice X = {} Slice Y = {}".format(sl[0][0], sl[0][1]))

        # ── ratio == 3: three merged segments ────────────────────────────────
        elif ratio == 3.0:
            print("INFO: Three Segments {}.".format(ratio))
            print("INFO: Slice X = {} Slice Y = {}".format(sl[0][0], sl[0][1]))

            third = (sl[0][1].stop - sl[0][1].start) // 3
            bounds = [
                (sl[0][1].start,           sl[0][1].start + third,     label),
                (sl[0][1].start + third,   sl[0][1].start + 2 * third, None),
                (sl[0][1].start + 2*third, sl[0][1].stop,              None),
            ]

            for k, (y_start, y_stop, seg_label) in enumerate(bounds):
                if seg_label is None:
                    max_label += 1
                    seg_label  = max_label

                segment_dict = {
                    'line':    line_number,
                    'label':   seg_label,
                    'start_x': sl[0][0].start,
                    'stop_x':  sl[0][0].stop,
                    'start_y': y_start,
                    'stop_y':  y_stop,
                }
                seg = labeled_line[
                    sl[0][0].start:sl[0][0].stop,
                    y_start:y_stop
                ]
                seg = np.where(seg == label, 255, 0)

                _plot_seg(
                    seg,
                    "line: {} segment: {} (3-split {})".format(line_number, seg_label, k + 1),
                    verbose, plot_callback
                )

                ws, ls, xs, ys = get_values_from_img(seg)
                segment_dict.update({
                    'wseg':     ws,
                    'lseg':     ls,
                    'xseg':     xs,
                    'yseg':     ys,
                    'baseline': np.argmax(np.std(seg, axis=1)),
                })
                line_dict['curves'].append(segment_dict)
                print("INFO: label: {}  length {}".format(seg_label, ls))

        # ── ratio == 2: two merged segments ──────────────────────────────────
        elif ratio == 2.0:
            print("INFO: Two Segments {} Slice X = {} Slice Y = {}".format(
                ratio, sl[0][0], sl[0][1]))

            half   = (sl[0][1].stop - sl[0][1].start) // 2
            bounds = [
                (sl[0][1].start,        sl[0][1].start + half, label),
                (sl[0][1].start + half, sl[0][1].stop,         None),
            ]

            for k, (y_start, y_stop, seg_label) in enumerate(bounds):
                if seg_label is None:
                    max_label += 1
                    seg_label  = max_label

                segment_dict = {
                    'line':    line_number,
                    'label':   seg_label,
                    'start_x': sl[0][0].start,
                    'stop_x':  sl[0][0].stop,
                    'start_y': y_start,
                    'stop_y':  y_stop,
                }
                seg = labeled_line[
                    sl[0][0].start:sl[0][0].stop,
                    y_start:y_stop
                ]
                seg = np.where(seg == label, 255, 0)

                _plot_seg(
                    seg,
                    "line: {} segment: {} (2-split {})".format(line_number, seg_label, k + 1),
                    verbose, plot_callback
                )

                ws, ls, xs, ys = get_values_from_img(seg)
                segment_dict.update({
                    'wseg':     ws,
                    'lseg':     ls,
                    'xseg':     xs,
                    'yseg':     ys,
                    'baseline': np.argmax(np.std(seg, axis=1)),
                })
                line_dict['curves'].append(segment_dict)
                print("INFO: label: {}  length {}".format(seg_label, ls))

        # ── ratio == 1: single segment ────────────────────────────────────────
        elif ratio == 1.0:
            print("INFO: One Segment {}.".format(ratio))
            print("INFO: Slice X = {} Slice Y = {}".format(sl[0][0], sl[0][1]))

            segment_dict = {
                'line':    line_number,
                'label':   label,
                'start_x': sl[0][0].start,
                'stop_x':  sl[0][0].stop,
                'start_y': sl[0][1].start,
                'stop_y':  sl[0][1].stop,
            }
            seg = labeled_line[
                sl[0][0].start:sl[0][0].stop,
                sl[0][1].start:sl[0][1].stop
            ]
            seg = np.where(seg == label, 255, 0)

            _plot_seg(
                seg,
                "line: {} segment: {}".format(line_number, label),
                verbose, plot_callback
            )

            ws, ls, xs, ys = get_values_from_img(seg)
            segment_dict.update({
                'wseg':              ws,
                'lseg':              ls,
                'xseg':              xs,
                'yseg':              ys,
                'firstpixel_abs_y':  np.argmax(seg[:, 0]) + segment_dict['start_x'] + ys[0],
                'baseline':          np.argmax(np.std(seg, axis=1)),
            })
            print("INFO: label: {}  length {}".format(label, ls))
            line_dict['curves'].append(segment_dict)

        elif ratio < 1.0:
            continue   # garbage — skip

    # ── Sort segments left-to-right and assign lead names ────────────────────
    line_dict['curves'] = sorted(line_dict['curves'], key=lambda d: d['start_y'])
    for i, d in enumerate(line_dict['curves']):
        d['name'] = line_leads[i]

    return line_dict


# =============================================================================
# ECG PLOTTING
# =============================================================================

def plot_ecg_signal(time, signal, ax):
    """
    Plot a single ECG lead onto an existing Axes with clinical-style grid.

    Returns the Axes object.
    """
    ax.plot(time, signal)
    min_t      = int(np.min(time))
    max_t      = round(np.max(time))
    major_ticks = np.arange(min_t, max_t + 1)
    ax.set_xticks(major_ticks)
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-',  color='red',   linewidth=1.0)
    ax.grid(which='minor', linestyle=':',  color='black', linewidth=0.5)
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Amplitude')
    return ax


def plot_ecg(df, columns, title,
             n_rows=4, n_columns=4, fs=500, figure_size=(20, 12),
             plot_callback=None):
    """
    Plot all ECG leads in a grid layout.

    Parameters
    ----------
    df            : pandas DataFrame — one column per lead
    columns       : iterable of column names to plot
    title         : figure title (usually the CSV filename)
    n_rows        : int
    n_columns     : int
    fs            : sampling frequency in Hz
    figure_size   : tuple (width, height) in inches
    plot_callback : callable(fig) | None  — GUI routing function
    """
    if (n_rows * n_columns) < len(columns):
        raise ValueError(
            'n_rows × n_columns must be >= number of columns to plot.'
        )

    fig, axes = plt.subplots(n_rows, n_columns, figsize=figure_size)
    fig.suptitle(title, fontsize=20)

    for index, col in enumerate(columns):
        if n_rows == 1 or n_columns == 1:
            ax = axes[index]
        else:
            row_index = int(index / n_columns)
            col_index = int(index - n_columns * row_index)
            ax = axes[row_index][col_index]

        signal = df[col]
        ts     = np.arange(signal.size) / fs
        ax     = plot_ecg_signal(ts, signal, ax)
        ax.set_title(col)

    fig.subplots_adjust(top=0.92, hspace=0.45, wspace=0.5)
    _show(fig, plot_callback)
    return fig
