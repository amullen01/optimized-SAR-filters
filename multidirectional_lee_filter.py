from math import exp
import numpy as np
import numba as nb
import math
from typing import Union, Tuple, List, Optional
import argparse
import sys
from pathlib import Path
import rioxarray as rxr

@nb.jit
def assert_window_size(win_size: int) -> None:
    """
    Asserts valid window size.
    Window size must be odd and bigger than 3.
    """
    assert win_size >= 3, 'ERROR: win size must be at least 3'

    if win_size % 2 == 0:
        print('use odd window size')

@nb.jit
def euclidean_distance(y1: int, x1: int, y2: int, x2: int) -> float:
    """
    Calculate Euclidean distance between two points.
    """
    return np.sqrt((y2 - y1)**2 + (x2 - x1)**2)

@nb.jit
def pad_nodata(arr: np.ndarray, nodata: float, buffer_distance: int) -> np.ndarray:
    """
    Pad nodata regions within bounds of image array using nearest valid neighbor within defined buffer radius
    """
    result = arr.copy()
    
    rows, cols = result.shape
    for i in range(rows):
        for j in range(cols):
            if result[i, j] == nodata or np.isnan(result[i, j]):

                found_value = nodata
                min_dist = np.inf
                # Check connected neighbors
                for dy in range(-buffer_distance, buffer_distance):
                    for dx in range(-buffer_distance, buffer_distance):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = i + dy, j + dx
                        if (0 <= ny < rows and 0 <= nx < cols):
                            dist = euclidean_distance(i, j, ny, nx)
                            if dist < min_dist and (arr[ny, nx] != nodata and ~np.isnan(arr[ny, nx])):
                                min_dist = dist
                                found_value = arr[ny, nx]
                
                result[i, j] = found_value
    
    return result

@nb.jit
def assert_indices_in_range(height: int, width: int, yup: int, ydown: int, 
                           xleft: int, xright: int) -> None:
    """
    Asserts index within image range.
    """
    assert xleft >= 0 and xleft <= width

    assert xright >= 0 and xright <= width

    assert yup >= 0 and yup <= height

    assert ydown >= 0 and ydown <= height

@nb.jit
def bresenham_line(kernel: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> None:
    """
    Bresenham's line algorithm
    """
    dx = abs(x2 - x1)
    dy = -abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx + dy
    
    current_x, current_y = x1, y1
    while True:
        kernel[current_y, current_x] = 1
        if current_x == x2 and current_y == y2:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            current_x += sx
        if e2 <= dx:
            err += dx
            current_y += sy

@nb.jit
def create_directional_kernels(ymin: int, ymax: int, xmin: int, xmax: int) -> np.ndarray:
    """
    create multidirectional kernels given index extents
    """

    height = ymax - ymin
    width = xmax - xmin
    
    # Pre-allocate array for all kernels
    boundary_count = height + width - 2  # Total boundary pixels
    kernels = np.zeros((boundary_count, height, width), dtype=np.int8)
    
    kernel_idx = 0
    
    # Generate kernels for top and bottom edges
    for x in range(width):
        # line from top to bottom edge
        kernel = np.zeros((height, width), dtype=np.int8)
        bresenham_line(kernel, x, 0, width-x-1, height-1)
        kernels[kernel_idx] = kernel
        kernel_idx += 1
    
    # Generate kernels for left and right edges (excluding corners)
    for y in range(1, height-1):
        # line from left to right edge
        kernel = np.zeros((height, width), dtype=np.int8)
        bresenham_line(kernel, 0, y, width-1, height-y-1)
        kernels[kernel_idx] = kernel
        kernel_idx += 1
    
    return kernels

@nb.jit
def mask_window_with_kernels(window: np.ndarray, kernels: np.ndarray) -> List[np.ndarray]:
    """
    Applies multiple directional kernels to a window and returns masked pixels for each kernel.
    
    Parameters:
    -----------
    window : 2D array (float32/64)
        Image window to process
    kernels : list of 2D arrays
        List of directional kernels (each with same shape as window)
    
    Returns:
    --------
    list of 1D arrays
        Pixel values for each kernel's active positions (1s)
    """
    results = []
    for kernel in kernels:
        pixels = np.zeros(int(np.sum(kernel)), dtype=window.dtype)
        idx = 0
        for i in range(window.shape[0]):
            for j in range(window.shape[1]):
                if kernel[i, j] > 0:
                    pixels[idx] = window[i, j]
                    idx += 1
        results.append(pixels)
    return results

@nb.jit
def dilate_kernels(kernels: np.ndarray) -> np.ndarray:
    """
    Dilate binary kernels (0s and 1s) with a value of 0.5 using a 3x3 window.
    
    Parameters:
    kernels : numpy.ndarray
        3D array of shape (n_kernels, height, width) containing binary kernels
        
    Returns:
    numpy.ndarray
        Dilated kernels with original 1s preserved and new 0.5 values at dilation boundaries
    """
    # Create output array
    dilated = np.zeros_like(kernels, dtype=np.float32)
    n_kernels, height, width = kernels.shape
    
    # Define 3x3 neighborhood offsets
    offsets = [(-1,-1), (-1,0), (-1,1),
               (0,-1),          (0,1),
               (1,-1),  (1,0), (1,1)]
    
    for k in range(n_kernels):
        for i in range(height):
            for j in range(width):
                if kernels[k, i, j] == 1:
                    # Preserve original 1s
                    dilated[k, i, j] = 1
                    # Dilate with 0.5 in 3x3 neighborhood
                    for di, dj in offsets:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < height and 0 <= nj < width:
                            if kernels[k, ni, nj] == 0:
                                dilated[k, ni, nj] = 0.5
    return dilated

@nb.jit
def calculate_weighted_stats(window: np.ndarray, weights: np.ndarray) -> Tuple[float, float]:
    """
    Numba-compatible calculation of weighted mean, and stddev.
    
    Parameters:
    window : numpy.ndarray
        The input values (same shape as weights)
    weights : numpy.ndarray
        The weights for each value in window
        
    Returns:
    tuple: (mean, stddev)
    """
    total_weight = np.sum(weights)
    
    # Calculate weighted mean
    mean = 0.0
    for i in range(window.size):
        mean += window.flat[i] * weights.flat[i]
    mean /= total_weight
    
    # Calculate weighted variance
    variance = 0.0
    for i in range(window.size):
        diff = window.flat[i] - mean
        variance += weights.flat[i] * (diff * diff)
    variance /= total_weight
    
    # Calculate standard deviation
    stddev_val = np.sqrt(variance)
    
    return mean, stddev_val

@nb.jit
def multidirectional_lee_filter(img_data: np.ndarray, win_size: int = 9, 
                               noise_abs: float = None, 
                               noise_rel: float = 1.0, 
                               nodata: Union[int, float] = np.nan) -> np.ndarray:
    """
    Apply multi-directional Enhanced Lee filter to reduce speckle noise while preserving edges.
    
    The multi-directional Lee filter analyzes image statistics along multiple directions
    to better preserve edges and linear features. It uses directional kernels to compute
    statistics and selects the direction with minimum variance, which typically corresponds
    to the edge direction.
    
    Parameters
    ----------
    img_data : np.ndarray
        Input 2D image array to be filtered. Can be any numeric type.
        Typically represents radar/SAR imagery with speckle noise.
    win_size : int, default=9
        Size of the sliding window (both width and height). Must be odd and >= 3.
        Larger windows provide more smoothing but may blur fine details.
    noise_abs : float, default=None
        Absolute noise threshold. If False, not used. If float, pixels with
        standard deviation > noise_abs are considered noisy and filtered.
    noise_rel : float, default=1
        Relative noise threshold as a multiplier of the mean image standard deviation.
        If False, not used. If float, determines the noise level for filtering.
    nodata : Union[int, float], default=0
        Value representing missing/invalid data. These pixels are preserved
        in the output without filtering.
        
    Returns
    -------
    np.ndarray
        Filtered image array with same shape and original dtype as input.
        Speckle noise is reduced while edges and linear features are preserved.
        
    Notes
    -----
    - Uses directional analysis to preserve edges better than standard Lee filter
    - Automatically pads nodata regions before filtering to avoid edge artifacts
    - Applies two-stage filtering: first absolute noise threshold, then relative
    - Preserves original data type and nodata values
    - Optimized with Numba for performance on large images
    
    Algorithm Steps
    ---------------
    1. Validate window size and convert to float64 for processing
    2. Pad nodata regions using nearest neighbor interpolation
    3. For each pixel:
       a. Extract window around pixel
       b. Create directional kernels for the window
       c. Dilate kernels to create weighted masks
       d. Calculate weighted statistics for each direction
       e. Select direction with minimum standard deviation
       f. Apply filtering based on noise thresholds
    4. Convert back to original data type
    
    Examples
    --------
    >>> # Basic usage with default parameters
    >>> filtered = multidirectional_lee_filter(noisy_image)
    
    >>> # With custom window size and noise parameters
    >>> filtered = multidirectional_lee_filter(
    ...     noisy_image, 
    ...     win_size=5, 
    ...     noise_abs=0.5, 
    ...     noise_rel=1.2,
    ...     nodata=-999
    ... )
    
    References
    ----------
    Based on the multidirectional Lee filter from SAGA GIS: 
    https://saga-gis.sourceforge.io/saga_tool_doc/7.6.2/grid_filter_3.html

    Lee, J.S. (1980): Digital image enhancement and noise filtering by use of local statistics. 
        IEEE Transactions on Pattern Analysis and Machine Intelligence, PAMI-2: 165-168.
    Lee, J.S., Papathanassiou, K.P., Ainsworth, T.L., Grunes, M.R., Reigber, A. (1998): A New 
        Technique for Noise Filtering of SAR Interferometric Phase Images. IEEE Transactions on 
        Geosciences and Remote Sensing 36(5): 1456-1465.
    Selige, T., BÃ¶hner, J., Ringeler, A. (2006): Processing of SRTM X-SAR Data to correct 
        interferometric elevation models for land surface process applications. In: BÃ¶hner, J., 
        McCloy, K.R., Strobl, J. [Eds.]: SAGA - Analysis and Modelling Applications. GÃ¶ttinger 
        Geographische Abhandlungen, Vol. 115: 97-104. 
        http://downloads.sourceforge.net/saga-gis/gga115_09.pdf.
    """

    assert_window_size(win_size)

    orig_dtype = img_data.dtype
    img_data = img_data.astype(np.float64)

    img_filtered = np.zeros_like(img_data)
    rows, cols = img_data.shape
    win_offset = ((win_size-1) / 2)

    #pad nodata
    img_data_padded = pad_nodata(img_data, nodata, win_offset)

    stds = np.zeros_like(img_data)
  
    for y in range(0, rows):
        yup = y - win_offset
        ydown = y + win_offset

        if yup < 0:
            yup = 0
        if ydown >= rows:
            ydown = rows

        for x in range(0, cols):

            if img_data[y, x] == nodata or np.isnan(img_data[y, x]):
                img_filtered[y, x] = nodata
                continue

            xleft = x - win_offset
            xright = x + win_offset

            if xleft < 0:
                xleft = 0
            if xright >= cols:
                xright = cols

            xleft = int(xleft)
            xright = int(xright)
            yup = int(yup)
            ydown = int(ydown)

            assert_indices_in_range(rows, cols, yup, ydown, xleft, xright)

            pix_value = img_data_padded[y, x]
            window = img_data_padded[yup:ydown+1, xleft:xright+1]

            #for multidirectional calculations
            kernels = create_directional_kernels(yup, ydown+1, xleft, xright+1)
            kernels = dilate_kernels(kernels)

            min_std = np.inf
            min_mean = np.inf

            for i in range(len(kernels)):
                mean, std = calculate_weighted_stats(window, kernels[i])

                if std < min_std:

                    min_std = std
                    min_mean = mean

                    if std==0:
                        stds[y,x]=np.nan
                    else:
                        stds[y,x] = std
            
            if not noise_abs is None and (min_std > noise_abs):
                b = (min_std**2 - noise_abs**2) / min_std**2
                img_filtered[y, x] = pix_value * b + (1 - b) * min_mean

            elif min_std > 0:
                img_filtered[y, x] = min_mean
            else:
                img_filtered[y, x] = pix_value

    if not noise_rel is None:
            mean_stddev = np.nanmean(stds)
            noise = noise_rel * mean_stddev
            
            for y in range(0, rows):
                for x in range(0, cols):
                    if img_data[y, x] == nodata:
                        continue
                    if np.isnan(img_filtered[y, x]):
                        continue
                    
                    b_val = stds[y, x]
                    if b_val > noise:
                        b = (b_val**2 - noise**2) / b_val**2
                        img_filtered[y, x] = img_data_padded[y, x] * b + (1 - b) * img_filtered[y, x]
            
    return img_filtered.astype(orig_dtype)

def main():
    """
    Main function for CLI. At minimum, specify path to input geotiff (-i) and output geotiff (-o). Defaults to win-size=9,
    noise-abs=None, noise-rel=1, and will try to process all bands in image assuming bands-first shape (band in 0th dimension).
    Increasing window size increases smoothing, and decreasing noise-rel and noise-abs increases the number of smoothed pixels.
    Experiment with different window sizes first, followed by noise-rel, and noise-abs to obtain desired filter performance.
    

    python multidirectional_lee_filter.py -i=filtered_images/S1_example.tif -o=filtered_images/S1_example_filt.tif \
    --verbose --win-size=7 
    """
    parser = argparse.ArgumentParser(
        description="Multi-directional Lee Filter for SAR Images - Advanced speckle reduction while preserving edges and linear features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '-i',
        type=str,
        help='Path to input SAR image file (e.g., .tif format)'
    )
    
    parser.add_argument(
        '-o', 
        type=str,
        help='Path for output filtered image file'
    )
    
    # Optional filter parameters
    parser.add_argument(
        '--win-size',
        type=int,
        default=9,
        help='Filter window size (NxN). Larger values provide more smoothing'
    )
    
    parser.add_argument(
        '--noise-abs',
        type=float,
        default=None,
        help='Absolute noise threshold. If None, uses relative threshold only'
    )
    
    parser.add_argument(
        '--noise-rel',
        type=float,
        default=1.0,
        help='Relative noise threshold multiplier'
    )
    
    parser.add_argument(
        '--nodata',
        type=float,
        default=np.nan,
        help='No-data value to mask during processing'
    )
    
    parser.add_argument(
        '--band-index',
        type=int,
        nargs='+',
        default=None,
        help='Band index(es) to process (0-based indexing). Can specify multiple bands: --band-index 0 1 2'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Convert paths to Path objects for better handling
    input_path = Path(args.i)
    output_path = Path(args.o)
    
    if args.verbose:
        print("=== Multi-directional Lee Filter for SAR Images ===")
        print("Advanced speckle reduction while preserving edges and linear features.")
        print()
    
    # Validate input file exists
    if not input_path.exists():
        print(f"Error: Input file does not exist: {input_path}")
        sys.exit(1)
    
    # Process the SAR image
    try:
        if args.verbose:
            print(f"Loading SAR image: {input_path}")
        
        # Load the SAR image
        im = rxr.open_rasterio(input_path)
        
        if args.verbose:
            print(f"Image shape: {im.shape}, Data type: {im.dtype}")
            print(f"Coordinate system: {im.rio.crs}")
        
        if args.band_index is None:
            args.band_index = np.arange(im.shape[0])

        # Validate band indices
        for band_idx in args.band_index:
            if band_idx >= im.shape[0]:
                raise ValueError(f"Band index {band_idx} out of range. Image has {im.shape[0]} bands.")
        
        if np.isnan(args.nodata):
            if not im.rio.nodata is None and np.isnan(args.nodata):
                print(f'warning: nodata from image metadata is {im.rio.nodata}, but filter is using np.nan')

        if (not args.noise_abs is None and args.noise_abs < 0) or (not args.noise_rel is None and args.noise_rel < 0):
            raise ValueError(f"noise_abs and noise_rel must be greater than zero")
        
        # Create a deep copy to avoid modifying original data
        im_filtered = im.copy(deep=True)
        
        if args.verbose:
            print("Applying multi-directional Lee filter...")
            print(f"Parameters: win_size={args.win_size}, noise_abs={args.noise_abs}, "
                  f"noise_rel={args.noise_rel}, nodata={args.nodata}")
            
        for band_idx in args.band_index:

            if args.verbose:
                print(f"Processing band {band_idx}...")

            # Apply the filter to the specified band
            im_filtered.values[band_idx] = multidirectional_lee_filter(
                im_filtered.values[band_idx],
                win_size=args.win_size,
                noise_abs=args.noise_abs,
                noise_rel=args.noise_rel,
                nodata=args.nodata
            )
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if args.verbose:
            print(f"Saving filtered image: {output_path}")
        
        # Save the filtered image with all metadata preserved
        im_filtered.rio.to_raster(str(output_path))
        
        if args.verbose:
            print("Processing completed successfully!")
            print(f"Input file: {input_path}")
            print(f"Output file: {output_path}")
            print()
            print("=== Processing Complete ===")
        else:
            print(f"Successfully processed: {input_path} -> {output_path}")
            
    except ImportError as e:
        print(f"Error: Required package not found. Please install required packages: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()