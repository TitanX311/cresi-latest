#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 07:46:39 2019

@author: avanetten
"""

import os
import sys
import cv2
import subprocess
# from osgeo import gdal
import rasterio
from rasterio.enums import ColorInterp
import numpy as np
import argparse
from multiprocessing.pool import Pool

# numbers retrieved from all_dems_min_max.py, and give min, max value for 
# each band over the entirety of SN3
rescale = {
    'tot_3band': {
        1: [63,  1178],
        2: [158, 1285],
        3: [148, 880]
    },
    # RGB corresponds to bands: 5, 3, 2
    'tot_8band': {
            1: [154, 669], 
            2: [122, 1061], 
            3: [119, 1520], 
            4: [62, 1497], 
            5: [20, 1342], 
            6: [36, 1505], 
            7: [17, 1853], 
            8: [7, 1559]}
}


###############################################################################
def convert_to_8Bit(inputRaster, outputRaster,
                    outputPixType="Byte",
                    outputFormat="GTiff",
                    rescale_type="perc",
                    percentiles=[2, 98],
                    band_order=[],
                    nodata_val=0,
                    max_zero_frac=0.3,
                    ):
    '''
    Convert 16bit image to 8bit
    rescale_type = [clip, perc, <dict>key
        if clip, scaling is done sctricly between 0 65535
        if rescale, each band is rescaled to a min and max
        set by percentiles
        if dict, access the 'rescale' dict at the beginning for rescaling
    percentiles, if using rescale_type=rescale, otherwise ignored
    if the array has greater than max_zero_frac == 0, then skip

    band_order determines which bands and in what order to create them.
        If band_order == [], use all bands.
        for WV3 8-band,  RGB corresponds to bands: 5, 3, 2
    https://gdal.org/programs/gdal_translate.html
    '''

    with rasterio.open(inputRaster) as src:
            # Determine bands to read (Rasterio is 1-indexed)
            if len(band_order) == 0:
                band_list = list(range(1, src.count + 1))
            else:
                band_list = band_order

            nbands = len(band_list)
            
            # Read selected bands into a single numpy array: shape (nbands, height, width)
            # Note: we read using band indexes directly
            img_data = src.read(band_list)
            
            out_bands_data = []

            # Iterate through each selected band to scale it
            for j, bandId in enumerate(band_list):
                band_arr = img_data[j].astype(np.float32)
                band_arr_flat = band_arr.flatten()

                if rescale_type == "perc":
                    band_arr_pos = band_arr_flat[band_arr_flat > 0]

                    # test zero fraction
                    if len(band_arr_flat) > 0:
                        zero_frac = 1.0 - (len(band_arr_pos) / (1.0 * len(band_arr_flat)))
                    else:
                        zero_frac = 1.0

                    if zero_frac >= max_zero_frac:
                        print(f"zero_frac = {zero_frac} for {inputRaster}, skipping...")
                        return "skipped_too_many_zeros"

                    if len(band_arr_pos) == 0:
                        bmin, bmax = np.min(band_arr_flat), np.max(band_arr_flat)
                    else:
                        bmin = np.percentile(band_arr_pos, percentiles[0])
                        bmax = np.percentile(band_arr_pos, percentiles[1])
                elif rescale_type == 'clip':
                    bmin, bmax = 0, 65535
                else: 
                    bmin, bmax = rescale[rescale_type][bandId]

                # Ensure bmin is 1 or greater if that logic is needed
                bmin = max(1, bmin)
                print(f"Band {bandId} -> bmin: {bmin}, bmax: {bmax}")

                # Avoid division by zero if min == max
                if bmax == bmin:
                    bmax += 1e-5

                # Scale to 0-255 range
                scaled = (band_arr - bmin) * (255.0 / (bmax - bmin))
                scaled = np.clip(scaled, 0, 255).astype(np.uint8)
                out_bands_data.append(scaled)

            # Stack bands back together
            out_data = np.stack(out_bands_data, axis=0)

            # Update metadata for 8-bit output
            meta = src.meta.copy()
            meta.update({
                'dtype': 'uint8',
                'count': nbands,
                'nodata': nodata_val,
                'driver': 'GTiff'
            })

            # Write out the new 8-bit image
            with rasterio.open(outputRaster, 'w', **meta) as dst:
                dst.write(out_data)
                # Set RGB color interpretation if output is 3 bands
                if nbands == 3:
                    dst.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue]

    return f"Successfully processed {inputRaster}"


###############################################################################
def gamma_correction(image, gamma=1.66):
    '''https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/'''
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
 
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


###############################################################################
# def calc_rescale(im_file_raw, m, percentiles):
#     srcRaster = gdal.Open(im_file_raw)
#     for band in range(1, 4):
#         b = srcRaster.GetRasterBand(band)
#         band_arr_tmp = b.ReadAsArray()
#         bmin = np.percentile(band_arr_tmp.flatten(),
#                              percentiles[0])
#         bmax= np.percentile(band_arr_tmp.flatten(),
#                             percentiles[1])
#         m[band].append((bmin, bmax))

#     return m


###############################################################################
def process_image(params):
    im_file, im_file_raw, im_file_out, \
    outputPixType, outputFormat, rescale_type, percentiles, \
    max_zero_frac, band_order \
    = params

    if not im_file.endswith('.tif'):
        return

    if not os.path.isfile(im_file_out):
        #apls_tools.convert_to_8Bit(im_file_raw, im_file_out,
        # print ("isinstance(rescale_type, dict):", isinstance(rescale[rescale_type], dict))
        cmd_str = convert_to_8Bit(im_file_raw, im_file_out,
                                   outputPixType=outputPixType,
                                   outputFormat=outputFormat,
                                   rescale_type=rescale_type,
                                   percentiles=percentiles,
                                   band_order=band_order,
                                   max_zero_frac=max_zero_frac)
    else:
        print ("File exists, skipping!", im_file_out)

    
###############################################################################
def dir_to_8bit(path_images_raw, path_images_8bit,
                command_file_loc='',
                outputPixType="Byte",
                outputFormat="GTiff",
                rescale_type="perc",
                percentiles=[2, 98],
                max_zero_frac=0.3,
                band_order=[],
                n_threads=12):
    '''Create directory of 8bit images'''

    # iterate through images, convert to 8-bit, and create masks
    im_files = [z for z in sorted(os.listdir(path_images_raw)) if z.endswith('.tif')]
    print("im_files:", im_files)

    params = []
    for i, im_file in enumerate(im_files):
           
        # create 8-bit image
        im_file_raw = os.path.join(path_images_raw, im_file)
        im_file_out = os.path.join(path_images_8bit, im_file)

        params.append((im_file, im_file_raw, im_file_out, \
                        outputPixType, outputFormat, rescale_type, percentiles, \
                        max_zero_frac, band_order))
                        
    pool = Pool(n_threads)
    pool.map(process_image, params)

    return

###############################################################################
def dir_to_8bit_single_threaded(path_images_raw, path_images_8bit,
                command_file_loc='',
                outputPixType="Byte",
                outputFormat="GTiff",
                rescale_type="perc",
                percentiles=[2, 98],
                max_zero_frac=0.3,
                band_order=[],
                n_threads=12):
    '''Create directory of 8bit images'''

    # os.makedirs(path_images_8bit, exist_ok=True)
    if len(command_file_loc) > 0:
        f = open(command_file_loc, 'w')

    # iterate through images, convert to 8-bit, and create masks
    im_files = sorted(os.listdir(path_images_raw))
    print("im_files:", im_files)

    # m = defaultdict(list)
    for i, im_file in enumerate(im_files):
    
        if not im_file.endswith('.tif'):
            continue

        if (i % 1) == 0:
            print("\n")
            print (i, im_file)
            
        # create 8-bit image
        im_file_raw = os.path.join(path_images_raw, im_file)
        im_file_out = os.path.join(path_images_8bit, im_file)
        #im_file_out = os.path.join(path_images_8bit, test_data_name + name_root + '.tif')
        # convert to 8bit
        # m = calc_rescale(im_file_raw, m, percentiles=[2,98])
        # continue
        
        if not os.path.isfile(im_file_out):
            #apls_tools.convert_to_8Bit(im_file_raw, im_file_out,
            # print ("isinstance(rescale_type, dict):", isinstance(rescale[rescale_type], dict))
            cmd_str = convert_to_8Bit(im_file_raw, im_file_out,
                                       outputPixType=outputPixType,
                                       outputFormat=outputFormat,
                                       rescale_type=rescale_type,
                                       percentiles=percentiles,
                                       band_order=band_order,
                                       max_zero_frac=max_zero_frac)

            if len(command_file_loc) > 0:
                f.write(cmd_str + '\n')
        else:
            print ("File exists, skipping!", im_file_out)

    if len(command_file_loc) > 0:
        f.close()

    return


###############################################################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, default='')
    parser.add_argument('--outdir', type=str, default='')
    parser.add_argument('--command_file_loc', type=str, default='')
    parser.add_argument('--rescale_type', type=str, default='perc',
                        help="clip, perc, tot_8band, tot_3band")
    parser.add_argument('--band_order', type=str, default='5,3,2',
                        help="',' separated list "
                        " set to '' to use default band order, 1-indexed")
    parser.add_argument('--percentiles', type=str, default='2,98',
                        help="',' separated list of min,max percentiles")
    parser.add_argument('--max_zero_frac', type=float, default=0.3,
                        help="max percentage of image we allow to be null")
    parser.add_argument('--n_threads', type=int, default=12,
                           help="num threads for multiprocessing")
    args = parser.parse_args()

    # parse band_order
    if len(args.band_order) == 0:
        band_order = []
    else:
        band_order_str = args.band_order.split(',')
        band_order = [int(z) for z in band_order_str]
    percentiles = [int(z) for z in args.percentiles.split(',')]

    # values that should remain constant
    outputPixType = "Byte"
    outputFormat = "GTiff"

    os.makedirs(args.outdir, exist_ok=True)

    dir_to_8bit(args.indir, args.outdir,
                command_file_loc=args.command_file_loc,
                rescale_type=args.rescale_type,
                band_order=band_order,
                outputPixType=outputPixType,
                outputFormat=outputFormat,
                percentiles=percentiles,
                max_zero_frac=args.max_zero_frac,
                n_threads=args.n_threads)
