#!/usr/bin/python

from math import ceil
from numpy import arange
from itertools import product
from rasterio import windows as rio_windows

def generate_windows(
    img_height, 
    img_width, 
    win_height, 
    win_width, 
    min_hoverlap, 
    min_woverlap, 
    boundless=False
):
    hc = ceil((img_height - min_hoverlap) / (win_height - min_hoverlap))
    wc = ceil((img_width - min_woverlap) / (win_width - min_woverlap))
    
    
    h_overlap = ((hc * win_height) - img_height) // (hc - 1)
    w_overlap = ((wc * win_height) - img_width) // (wc - 1)
    
    
    hslack_res = ((hc * win_height) - img_height) % (hc - 1)
    wslack_res = ((wc * win_width) - img_width) % (wc - 1)
    
    dh = win_height - h_overlap
    dw = win_width - w_overlap
    
    row_offsets = arange(0, (img_height-h_overlap), dh)
    col_offsets = arange(0, (img_width-w_overlap), dw)
    
    if hslack_res > 0:
        row_offsets[-hslack_res:] -= arange(1, (hslack_res + 1), 1)
    if wslack_res > 0:
        col_offsets[-wslack_res:] -= arange(1, (wslack_res + 1), 1)
    
    row_offsets = row_offsets.tolist()
    col_offsets = col_offsets.tolist()
    
    offsets = product(col_offsets, row_offsets)
    
    indices = product(range(len(col_offsets)), range(len(row_offsets)))
    
    big_window = rio_windows.Window(
        col_off=0, 
        row_off=0, 
        width=img_width, 
        height=img_height
    )
    
    for index, (col_off, row_off) in zip(indices, offsets):
        window = rio_windows.Window(
            col_off=col_off,
            row_off=row_off,
            width=win_width,
            height=win_height
        )
        if boundless:
            yield index, window
        else:
            yield index, window.intersection(big_window)
