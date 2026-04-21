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
from edt_utils import is_nan, py_blockproc, display_segments, detect_ref_pulse, print_line_dict,segment_to_df, remove_text
from ss import pattern_match
from edt_utils import process_line,get_values_from_img,measure_extract_pulse ,plot_ecg, extract_image
from scipy.signal import find_peaks
import operator

def ecg_to_csv(image_name, template_name, csv_name, config_dict):

    # BORDER_GAP = 2 # gap around the border
    # layout = (3,4)
    # # indicate if  there is a pulse
    # # pulse = 0 - pulse in all lines
    # # pulse = line_list - pulse present only on the the line llist
    # #
    # pulse = [0,1,2]  
    # rhythm = 4 # which line has the rhythum
    # verbose = 3
    # mmpsec = 25 # 25 mm/seg
    # mmpmv = 10 # 10 mm/mV
    # pulse_width_mm = 5 # pulse width in mm
    # pulse_height_mm =10  # pulse height in mm
    # pulse_per_sec = pulse_width_mm/mmpsec
    # pulse_per_mv= pulse_height_mm/mmpmv
    # sample_frequency = 500
    # time_lead = 2.5 # duratiom of the segment in seconds
    # num_sampling_points = time_lead/(1/sample_frequency)
    # location = 'right'

    layout = config_dict['layout']
    pulse  = config_dict['pulse']
    rhythm = config_dict['rhythm']
    verbose = config_dict['verbose']
    strategy = config_dict['strategy']
    thres_value = config_dict['thres_value']
    lower = config_dict['lower']
    upper = config_dict['upper']
    kSize2d = config_dict['kSized2d']
    kSize1d = config_dict['kSized1d']
    perc_space_leads = config_dict['perc_space_leads']
    dilation = config_dict['dilation']
    perc_max_dist = config_dict['perc_max_dist']

    # the names dependending on the layout
    if layout[1]== 4 and layout[0]==3:
        lt_leads = [ ['I', 'aVR','V1','V4'],
                        ['II','aVL','V2','V5'],
                        ['III','aVF', 'V3','V6'],
                        ['II']

                        ]
    elif layout[1]==2:
        raise NotImplementedError ('Not implemented' )
    elif layout[1]==1:

       # raise NotImplementedError ('Not implemented' )
       lt_leads = [ ['I'],
                    ['II'],
                    ['III'],
                    ['aVR'],
                    ['aVL'],
                    ['aVF'],
                    ['V1'],
                    ['V2'],
                    ['V3'],
                    ['V4'],
                    ['V5'],
                    ['V6']]

    else:
        raise ValueError('columns must be 4, 2 or 1')
    
    # Define pulse detection

    if pulse == 0 :
        print("INFO: No pulse to be detected")
    elif pulse == -1:
        print("INFO: pulse to be detected in all lines")
    elif  isinstance(pulse, list): 
        print("INFO: pulse on lines: {}.".format(pulse)) 
        for p in pulse:
            lt_leads[p].append('Pulse')
    elif isinstance(pulse, int): 
        print("INFO: pulse on line: {}.".format(pulse)) 
        lt_leads[pulse].append('Pulse')

    else:
        raise ValueError('pulse should  be 0, an int or a list')
    
    # Define rhythm

    if rhythm == 0:
        print("INFO: No rhythm lead") 
    else:
        print("INFO: rhythm on line : {}.".format(rhythm)) 


    # config_dict ={}
    # config_dict['layout']=layout   # tuple with the layout
    # config_dict['rhythm'] = rhythm # which row has the rhythm sig
    # config_dict['verbose'] = verbose # 
    # config_dict['pulse'] = pulse # which lines have pulse
    
    # config_dict['pulse_width_mm']  = pulse_width_mm
    # config_dict['pulse_height_mm'] = pulse_height_mm
    # config_dict['pulse_per_mv']= pulse_per_mv
    # config_dict['pulse_per_sec']= pulse_per_sec
    # config_dict['num_sampling_points']= num_sampling_points

    #load the image 

    #image_name = 'images/ecg_test.png'  # select image
    image = cv.imread(image_name)

    # sanity check
    if image is None:
        print('Cannot open image: ' + image_name)
        sys.exit(0)

    if verbose > 1:
        plt.imshow(image)
        print("INFO: Image Shape {}.".format(image.shape))

   
    if strategy =='color':
        img_hsv=cv.cvtColor(image, cv.COLOR_BGR2HSV)
        #Filter color to remove the grid
        #lower=(0,0,0) # black colssor
        #upper=(179,255,220) # dark gray
        mask = cv.inRange(img_hsv, lower, upper)
        result = img_hsv.copy()
        result[mask!=255] = (255, 255, 255) # if it is not very dark set it to white

        #Convert to gray scale
        image = cv.cvtColor(result, cv.COLOR_HSV2BGR )
        image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # To binary image
        ret, th1 = cv.threshold(image_gray, thres_value, 255,cv.THRESH_BINARY)


    if strategy == 'filter':
        image_gray = extract_image(image, kSize2d, kSize1d)
        ret, th1 = cv.threshold(image_gray, thres_value, 255,cv.THRESH_BINARY) # transform to binary

    if strategy == 'none':
        image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        ret, th1 = cv.threshold(image_gray, thres_value, 255,cv.THRESH_BINARY) # transform to binary

    if verbose > 0:
        plt.imshow(image_gray, cmap="gray")
        plt.show()
        print("INFO: gray scale image Shape {}.".format(image_gray.shape))



    if verbose > 2:
        plt.imshow(th1, cmap="gray")
        plt.show()
        print("INFO: Binary image Shape {}.".format(th1.shape))

    if dilation != 0:    
        foreground  = cv.morphologyEx(255-th1,cv.MORPH_DILATE,np.ones((3,3)),iterations=dilation)
    else:
        foreground = 255-th1


    # contours, _ = cv.findContours(foreground, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    # rectangular_contours = get_rectangular_contours(contours)

    # if verbose > 1:
    #     plt.imshow(foreground, cmap="gray")
    #     plt.show()

    # contour_image = image_gray.copy()

    # # find the biggest countour (c) by the area
    # c = max(contours, key = cv.contourArea)
    # x_border,y_border,w_border,h_border = cv.boundingRect(c)
    # # draw the biggest contour (c) in green
    # cv.rectangle(contour_image,(x_border,y_border),(x_border+w_border,y_border+h_border),(0,255,0),10)

    # if verbose > 1:
    #     plt.imshow(contour_image, cmap="gray")
    #     #TODO: add title
    #     plt.show()


    # # ECG image extracted from the main image
    

    # foreground  = 255-th1[y_border+BORDER_GAP:y_border+h_border-BORDER_GAP,
    #                     x_border+BORDER_GAP:x_border+w_border-BORDER_GAP]
    if verbose > 0:
        plt.imshow(foreground, cmap = "gray")
        plt.show()


    #template_name = 'images/pul.png'
    template = cv.imread(template_name, cv.IMREAD_GRAYSCALE)

    # sanity check
    if image is None:
        print('Cannot open the template: ' + template_name)
        new_template = None
    else:
        #load template to find the pulse
        _, new_template = cv.threshold(template, 127, 255, cv.THRESH_OTSU)
        new_template = (new_template != 255) * np.uint8(255)
        
    if verbose > 2:
        plt.imshow(new_template, cmap = "gray")
        plt.show()

    # Extract the individual leads (lines)

    temp= py_blockproc(foreground,(1,foreground.shape[1]), func=0)
    median_temp = np.median(temp.flatten())
    peak_indices, peak_dict = find_peaks(temp.flatten(), height=median_temp, distance=round(temp.flatten().size*perc_space_leads, 0))
    peak_heights = peak_dict['peak_heights']

    highest_peak_index = peak_indices[np.argsort(peak_heights)]



    if verbose > 0 :
        plt.plot(temp.flatten())
        # get the leads and the rhythm
        plt.plot(highest_peak_index[-(layout[0]+1):], temp[highest_peak_index[-(layout[0]+1):]], "x")
        plt.plot(median_temp*np.ones_like(temp), "--", color="gray")
        plt.show()

    # Calculate the distance between selected peaks

    ordered_hp_index = sorted(highest_peak_index[-(layout[0]+1):])


    peak_dist = [np.abs(t - s) for s, t in zip(ordered_hp_index, ordered_hp_index[1:])]
    max_dist = int(np.round(max(peak_dist)*perc_max_dist,0))

    # Cut the image according to the number of rows in the layout
    # slices_x is a list of tuples

    slices_x = [(max(0, s-max_dist), min(foreground.shape[0],s+max_dist),None) for s in ordered_hp_index]

    slices_y = [(0, foreground.shape[1], None) for s in ordered_hp_index]

    if verbose > 0 :
        print("INFO: slices: {}". format(slices_x))


    # Create a list to store the processed lines

    proc_line_list =[]

    h, w = foreground.shape
    blank_image =  np.zeros(shape=(h, w), dtype=np.uint8)

    # Extract and process the leads

    for i, slx in enumerate(slices_x): 
    
        line = foreground[slice(*slx),slice(*(0, foreground.shape[1], None))]
        offset = slx # reference to locate the segment in the image
        plt.imshow(line, cmap="gray")
        plt.show()
        structure = np.array([[1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1]], np.uint8)
    

        labeled_line, nb = ndimage.label(line, structure=structure)


        if verbose > 0:
            print("INFO: Number of segments {} on line {}.".format(nb, i))
            display_segments('Labeled line' + str(i), labeled_line)
        
        
        if (pulse == -1) or (i in pulse) :   # Check if the pulse is present
            line_signal = (labeled_line != 0) * np.uint8(255)
            #line_signal = np.where(labeled_line == 0, 0, 255)

            # plt.imshow(line_signal, cmap = "gray")

            #Try to detect the pulse
            line_copy = line_signal.copy()
            #line_copy = line_copy.astype("uint8")

            template_width, template_height = template.shape
            line_copy_width, line_copy_height = line_copy.shape
            _, _, xt, yt = get_values_from_img(new_template)
            wt, ht = measure_extract_pulse(xt, yt, verbose=0)
            config_dict['hpulse'] = ht #default values
            config_dict['wpulse'] = wt

            # pattern matching 
            # method = 'euclidean'
            # _,_,_,line_signal = get_values_from_img(line_copy)
            # _,_,_,template_signal = get_values_from_img(new_template)

            # #put the same baseline
            # baseline = np.argmax(np.std(line_copy, axis =1))


            # y_best = pattern_match(np.array(line_signal), np.array(template_signal+baseline),method)
            # print('DEBUG: pulse detected by template in line {} in {}'.format(i, y_best))


            # Pulse detection by template
            detected,location,  similarity_value, x,y, wpulse, hpulse= detect_ref_pulse(line_copy, new_template)
            print("INFO: line {}: best similarity value = {} in {}". format(i,similarity_value,y))

            if detected :
                if  location =='right':
                    sliced_labeled_line = labeled_line[:,0:y].copy()
                elif location == 'left':
                    sliced_labeled_line = labeled_line[:,y+int(wpulse):].copy()
                else:
                    sliced_labeled_line = labeled_line.copy()     
            else:
                 if is_nan(wpulse):
                    wpulse = wt
                    hpulse = ht
                 sliced_labeled_line = labeled_line.copy() 

           
                
            if verbose > 0:
                if detected:
                    print('INFO: pulse detected by template in line {} in {}'.format(i,y))
                    plt.imshow(line_copy[x:x+int(template_width)+1,y:y+int(template_height)+1], cmap ="gray")
                    plt.show()
                else:
                    print('INFO: pulse NOT detected by template in line {}'.format(i))
                    #sliced_labeled_line = labeled_line.copy() 
           
                

        else:
            print("INFO: line {} has no pulse to detect".format(i))
            wpulse = wt
            hpulse = ht

        # TODO: add info in config_dict  
        config_dict['wpulse']= wpulse
        config_dict['hpulse']= hpulse

        # Process line
        line_dict = process_line(i,sliced_labeled_line,offset,lt_leads[i], config_dict, config_dict['verbose'])
        proc_line_list.append(line_dict)


    #Print to check if everything is OK

    for i, line in enumerate(proc_line_list): 
        print("INFO: processing line {}".format(i))
        print_line_dict(line)

    #TODO remove the rhythm form the list of lines
    if config_dict['rhythm'] != 0:
        proc_line_list.pop(rhythm-1)
        
    # convert do a dataframe
    ecg_df= segment_to_df(proc_line_list, pulse_per_sec, pulse_per_mv,num_sampling_points)
    ecg_df.to_csv(csv_name)

    return ecg_df

# Main program 
filename = 'ecg1'
image_name = filename + '.png'
template_name = 'pul.png'

csv_name =filename + '.csv'
layout = (3,4)
pulse = [0,1,2]  
rhythm = 4 # which line has the rhythum
verbose = 1
mmpsec = 25 # 25 mm/seg
mmpmv = 10 # 10 mm/mV
pulse_width_mm = 5 # pulse width in mm
pulse_height_mm =10  # pulse height in mm
pulse_per_sec = pulse_width_mm/mmpsec
pulse_per_mv= pulse_height_mm/mmpmv
sample_frequency = 500
time_lead = 2.5 # duratiom of the segment in seconds
num_sampling_points = time_lead/(1/sample_frequency)
location = 'right'
strategy = 'none'  # It can be filter or color
lower=(0,0,0) # black color
upper=(179,255,220) # dark gray
thres_value = 127
kSize2d = 3 
kSize1d = 3
perc_space_leads =0.2
dilation = 10
perc_max_dist = 0.7 


config_dict ={}
config_dict['pulse'] = pulse # which lines have pulse
config_dict['rhythm'] = rhythm # which row has the rhythm signal
config_dict['verbose'] = verbose # 
config_dict['mmpsec']= mmpsec
config_dict['mmpmv']=mmpmv
config_dict['pulse_width_mm'] = pulse_width_mm
config_dict['pulse_height_mm'] = pulse_height_mm
config_dict['pulse_per_sec '] = pulse_per_sec
config_dict['pulse_per_mv'] = pulse_per_mv
config_dict['sample_frequency'] = sample_frequency
config_dict['time_lead'] = time_lead
config_dict['location']= location
config_dict['layout']=layout   # tuple with the layout    
config_dict['pulse_width_mm']  = pulse_width_mm
config_dict['pulse_height_mm'] = pulse_height_mm
config_dict['pulse_per_mv']= pulse_per_mv
config_dict['pulse_per_sec']= pulse_per_sec
config_dict['num_sampling_points']= num_sampling_points
config_dict['strategy'] = strategy
config_dict['lower']= lower
config_dict['upper']= upper
config_dict['thres_value'] = thres_value
config_dict['kSized2d'] = kSize2d
config_dict['kSized1d'] = kSize1d
config_dict['perc_space_leads'] = perc_space_leads
config_dict['dilation'] = dilation
config_dict['perc_max_dist'] = perc_max_dist 


df=ecg_to_csv(image_name ,template_name, csv_name, config_dict )
#df.plot(subplots=True, figsize=(12, 12)); plt.legend(loc='best');plt.show()

# Plot in the lay out

#plot_ecg(df,df.columns,csv_name, n_rows =layout[0] , n_columns = layout[1], figure_size = (20, 12))
plot_ecg(df,df.columns,csv_name, n_rows = layout[0], n_columns = layout[1], fs = 500, figure_size = (20, 12))
plt.show()

print("THE END")



    