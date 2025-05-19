import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import seaborn as sns
from configuration import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button
from mmwave.dataloader import DCA1000
from mmwave import dsp
from mmwave.dsp.utils import Window
import time
from datetime import datetime
from scipy import stats
import sys
plt.rcParams.update({'font.size': 12})
plt.rcParams["figure.figsize"] = (10, 7)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

fig = plt.figure(figsize=(18, 10))
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(133, projection='3d')

stop_flag = False
start_flag = True  
Tp = 14e-6
Tc = 72e-6

def init_dca():
    dca = DCA1000()
    return dca 

def collect_data(dca, num_frames=1):
    adc_data = dca.read(num_frames=int(num_frames))
    return adc_data

def stop_plot(event):
    global stop_flag, start_flag
    stop_flag = True
    start_flag = False

def start_plot(event):
    global stop_flag, start_flag
    stop_flag = False
    start_flag = True

def get_pcd(det_matrix, aoa_input):
    fft2d_sum = det_matrix.astype(np.int64)
    thresholdDoppler, noiseFloorDoppler = np.apply_along_axis(func1d=dsp.ca_,
                                                                axis=0,
                                                                arr=fft2d_sum.T,
                                                                l_bound=1.5,
                                                                guard_len=4,
                                                                noise_len=16)

    thresholdRange, noiseFloorRange = np.apply_along_axis(func1d=dsp.ca_,
                                                            axis=0,
                                                            arr=fft2d_sum,
                                                            l_bound=2.5,
                                                            guard_len=4,
                                                            noise_len=16)

    thresholdDoppler, noiseFloorDoppler = thresholdDoppler.T, noiseFloorDoppler.T
    det_doppler_mask = (det_matrix > thresholdDoppler)
    det_range_mask = (det_matrix > thresholdRange)

    # Get indices of detected peaks
    full_mask = (det_doppler_mask & det_range_mask)
    det_peaks_indices = np.argwhere(full_mask == True)

    # peakVals and SNR calculation
    peakVals = fft2d_sum[det_peaks_indices[:, 0], det_peaks_indices[:, 1]]
    snr = peakVals - noiseFloorRange[det_peaks_indices[:, 0], det_peaks_indices[:, 1]]

    dtype_location = '(' + str(numTxAntennas) + ',)<f4'
    dtype_detObj2D = np.dtype({'names': ['rangeIdx', 'dopplerIdx', 'peakVal', 'location', 'SNR'],
                                'formats': ['<i4', '<i4', '<f4', dtype_location, '<f4']})
    detObj2DRaw = np.zeros((det_peaks_indices.shape[0],), dtype=dtype_detObj2D)
    detObj2DRaw['rangeIdx'] = det_peaks_indices[:, 0].squeeze()
    detObj2DRaw['dopplerIdx'] = det_peaks_indices[:, 1].squeeze()
    detObj2DRaw['peakVal'] = peakVals.flatten()
    detObj2DRaw['SNR'] = snr.flatten()

    # Further peak pruning. This increases the point cloud density but helps avoid having too many detections around one object.
    detObj2DRaw = dsp.prune_to_peaks(detObj2DRaw, det_matrix, numDopplerBins, reserve_neighbor=True)

    # --- Peak Grouping
    detObj2D = dsp.peak_grouping_along_doppler(detObj2DRaw, det_matrix, numDopplerBins)
    SNRThresholds2 = np.array([[2, 23], [10, 11.5], [35, 16.0]])
    peakValThresholds2 = np.array([[4, 275], [1, 400], [500, 0]])
    detObj2D = dsp.range_based_pruning(detObj2D, SNRThresholds2, peakValThresholds2, numRangeBins, 0.5, range_resolution)

    azimuthInput = aoa_input[detObj2D['rangeIdx'], :, detObj2D['dopplerIdx']]

    Psi, Theta, Ranges, dopplers, xyzVec = dsp.beamforming_naive_mixed_xyz(azimuthInput, detObj2D['rangeIdx'], 
                                                                detObj2D['dopplerIdx'], 
                                                                range_resolution, method='Bartlett')

    return Ranges, dopplers, xyzVec


def iterative_range_bins_detection(rangeResult):
    rangeResult = np.transpose(np.stack([rangeResult[0::3], rangeResult[1::3], rangeResult[2::3]], axis=1),axes=(1,2,0,3))
    # print("rangeResult.shape = ", rangeResult.shape)
    range_result_absnormal_split=[]
    for i in range(numTxAntennas):
        for j in range(numRxAntennas):
            r_r=np.abs(rangeResult[i][j])
            #first 10 range bins i.e 40 cm make it zero
            r_r[:,0:10]=0
            min_val = np.min(r_r)
            max_val = np.max(r_r)
            r_r_normalise = (r_r - min_val) / (max_val - min_val) * (1000 - 0) + 0
            range_result_absnormal_split.append(r_r_normalise)
    
    range_abs_combined_nparray=np.zeros((numLoopsPerFrame,numADCSamples))
    
    # Average of all the Tx x Rx maps in absoluate sense
    for ele in range_result_absnormal_split:
        range_abs_combined_nparray+=ele
    range_abs_combined_nparray/=(numTxAntennas*numRxAntennas)
    
    # Average across chirps --> 256 length array
    range_abs_combined_nparray_collapsed=np.sum(range_abs_combined_nparray,axis=0)/numLoopsPerFrame

    # Sort in desceding order, take first 5 range bins
    peaks_min_intensity_threshold = np.argsort(range_abs_combined_nparray_collapsed)[::-1][:5]

    # Maximum intensity range bin
    max_range_index=np.argmax(range_abs_combined_nparray_collapsed)

    return max_range_index, peaks_min_intensity_threshold, rangeResult

def static_clusters(pointCloud, alpha): 
    std_dev_mult_factor = 1
    angle = np.arctan2(pointCloud[:,[0]], pointCloud[:,[1]])
    angle = np.where(angle > np.pi/2, angle - np.pi, angle)
    angle = np.where(angle < -np.pi/2, angle + np.pi, angle)
    vel_estimates = (pointCloud[:,[3]] / np.cos(alpha- angle)).T[0]
    kde = stats.gaussian_kde(vel_estimates)
    x_vals = np.linspace(min(vel_estimates), max(vel_estimates), 1000)
    pdf_vals = kde(x_vals)
    mode_kde = x_vals[np.argmax(pdf_vals)]
    cdf_vals = np.cumsum(pdf_vals)
    cdf_vals /= cdf_vals[-1] 
    mask = cdf_vals >= (1 - 0.95) 
    selected_x = x_vals[mask]
    selected_pdf = pdf_vals[mask]
    mean_kde = np.sum(selected_x * selected_pdf) / np.sum(selected_pdf)
    variance_kde = np.sum((selected_x - mean_kde) ** 2 * selected_pdf) / np.sum(selected_pdf)
    std_kde = np.sqrt(variance_kde)    
    static_mask = (mean_kde - std_dev_mult_factor * std_kde < vel_estimates) & (vel_estimates < mean_kde + std_dev_mult_factor * std_kde) 
    static_points = pointCloud[static_mask]
    dynamic_points = pointCloud[~static_mask]
    static_range_bins = static_points[:, 4]
    # static_range_bins = [np.sqrt(static_points[i][0]**2 + static_points[i][1]**2  + static_points[i][2]**2)/range_resolution for i in range(static_points.shape[0])]
    assert len(static_points)==len(static_range_bins), "Number of static points != Number of range bins"
    return static_points, static_range_bins


def calc_alpha(point1, point2):
    """
    Calcs alpha : angle between the radar Y axis and the velocity vector

    Args:
    point1 [x, y, z, V]
    point2 [x, y, z, V]
    """
    theta1 = np.arctan(point1[0] / point1[1])
    theta2 = np.arctan(point2[0] / point2[1])
    a = point2[3]*np.cos(theta1) - point1[3]*np.cos(theta2)
    b = point1[3]*np.sin(theta2) - point2[3]*np.sin(theta1)
    alpha = np.arctan(a/b)
    return alpha

def estimate_alpha(pointCloud):
    alpha_list = []
    for point1, point2 in itertools.combinations(pointCloud, 2):
        alpha_list.append(calc_alpha(point1, point2))
    alpha_list = np.array(alpha_list)
    
    # Generate histogram
    counts, bin_edges = np.histogram(alpha_list, bins=100, range=(alpha_list.min(), alpha_list.max()))

    # Find index of bin with the maximum count
    max_bin_index = np.argmax(counts)

    # Calculate the mode as the center of the bin with the highest frequency
    mode_estimate = (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1]) / 2

    return mode_estimate


def get_phase(r,i):
    if r==0:
        if i>0:
            phase=np.pi/2
        else :
            phase=3*np.pi/2
    elif r>0:
        if i>=0:
            phase=np.arctan(i/r)
        if i<0:
            phase=2*np.pi - np.arctan(-i/r)
    elif r<0:
        if i>=0:
            phase=np.pi - np.arctan(-i/r)
        else:
            phase=np.pi + np.arctan(i/r)
    return phase

def phase_unwrapping(phase_len,phase_cur_frame):
    i=1
    new_signal_phase = phase_cur_frame
    for k,ele in enumerate(new_signal_phase):
        if k==len(new_signal_phase)-1:
            continue
        if new_signal_phase[k+1] - new_signal_phase[k] > 1.5*np.pi:
            new_signal_phase[k+1:] = new_signal_phase[k+1:] - 2*np.pi*np.ones(len(new_signal_phase[k+1:]))
    return np.array(new_signal_phase)


def solve_equation(phase_cur_frame, xyz_vals, alpha):
    phase_diff=[]
    for j in range (1,len(phase_cur_frame)):
        phase_diff.append(phase_cur_frame[j]-phase_cur_frame[j-1])
    l = 4e-3
    c = l / (4*np.pi)
    X0 = xyz_vals[0]
    Y0 = xyz_vals[1]
    K = X0*np.sin(alpha) + Y0*np.cos(alpha)
    r0 = np.sqrt(X0**2 + Y0**2)

    roots_of_frame=[]
    for i,val in enumerate(phase_diff):
        Theta_t=(phase_diff[i]*0.001/3.14)/(3*(Tp+Tc))
        t=3*(i+1)*(Tp+Tc)

        #Coefficients
        a_t = t**2
        b_t = -2*K*t
        c_t = K**2 - (t**2)*(c**2)*(Theta_t**2)
        d_t = 2*t*K*(c**2)*(Theta_t**2)
        e_t = - (r0**2) * (c**2) * (Theta_t**2)
        coefficients=[a_t, b_t, c_t, d_t, e_t]
        root=min(np.abs(np.roots(coefficients)))  #Taking the min root 
        roots_of_frame.append(root)
    median_root=np.median(roots_of_frame)
    final_roots=[]
    for root in roots_of_frame:
        if root >0.9*median_root and root<1.1*median_root:
            final_roots.append(root)
    return np.mean(final_roots)


def get_velocity_antennawise(range_FFT_, range_bin, xyz_vals, alpha):
    phase_per_antenna=[]
    vel_peak=[]
    range_bin = int(range_bin)
    for k in range(0,numLoopsPerFrame):
        r = range_FFT_[k][range_bin].real
        i = range_FFT_[k][range_bin].imag
        phase=get_phase(r,i)
        phase_per_antenna.append(phase)
    phase_cur_frame=phase_unwrapping(len(phase_per_antenna),phase_per_antenna)
    cur_vel=solve_equation(phase_cur_frame, xyz_vals, alpha)
    return cur_vel


def dopplerFFT(rangeResult):  
    windowedBins2D = rangeResult * np.reshape(np.hamming(numLoopsPerFrame), (1, 1, -1, 1))
    dopplerFFTResult = np.fft.fft(windowedBins2D, axis=2)
    dopplerFFTResult = np.fft.fftshift(dopplerFFTResult, axes=2)
    return dopplerFFTResult

def get_velocity(rangeResult, range_bins, static_pcd, alpha):
    vel_array_frame=[]
    for k, range_bin in enumerate(range_bins):
        vel_arr_all_ant=[]
        for i in range(0,numTxAntennas):
            for j in range(0,numRxAntennas):
                xyz_vals = static_pcd[k]
                cur_velocity=get_velocity_antennawise(rangeResult[i][j], range_bin, xyz_vals, alpha)
                vel_arr_all_ant.append(cur_velocity)
        vel_array_frame.append(vel_arr_all_ant)
    return vel_array_frame


def speed_estimation_fn(range_bins, rangeResult, static_pcd, alpha):
    vel_array_frame = np.array(get_velocity(rangeResult,range_bins, static_pcd, alpha)).flatten()
    return vel_array_frame  

def update(frame_data):
    rangefft_out, det_matrix_vis, vel_mean, pointcloud, i = frame_data

    ax1.cla()
    ax2.cla()
    ax3.cla()

    sns.heatmap(rangefft_out / rangefft_out.max(), ax=ax1, cbar=False, cmap='viridis')
    ax1.set_title("Range Chirp Plot")
    ax1.set_xticks(np.linspace(0, rangefft_out.shape[1]-1, 10, dtype=int))  # 10 x-ticks
    ax1.set_yticks(np.linspace(0, rangefft_out.shape[0]-1, 10, dtype=int))  # 10 y-ticks
    ax1.set_xticklabels(np.linspace(0, rangefft_out.shape[1]-1, 10, dtype=int))
    ax1.set_yticklabels(np.linspace(0, rangefft_out.shape[0]-1, 10, dtype=int))
    sns.heatmap(det_matrix_vis / det_matrix_vis.max(), ax=ax2, cbar=False, cmap='viridis')
    ax2.set_title("Range-Doppler Plot")
    ax2.set_xticks(np.linspace(0, det_matrix_vis.shape[1]-1, 10, dtype=int))
    ax2.set_yticks(np.linspace(0, det_matrix_vis.shape[0]-1, 10, dtype=int))
    ax2.set_xticklabels(np.linspace(0, det_matrix_vis.shape[1]-1, 10, dtype=int))
    ax2.set_yticklabels(np.linspace(0, det_matrix_vis.shape[0]-1, 10, dtype=int))

    x, y, z = pointcloud[:,0], pointcloud[:,1], pointcloud[:,2]
    
    ax3.scatter(x, y, z, c='blue', marker='o', s=20)

    ax3.set_title("3D Pointcloud")
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_xlim(-5,5)
    ax3.set_ylim(0,10)
    ax3.set_zlim(-5,5)

    return ax1, ax2, ax3,


if __name__ == "__main__":
    # Array to store plot data
    frames_data = []
    i = 0 
    prev_range_bins = None
    overlapped_range_bins = []
    filename = "bot_2025-02-11_19_47_42_5objectsday1__only_sensor.bin"
    adc_data = np.fromfile(f'./dataset/{filename}', dtype=np.uint16) 
    adc_data = adc_data.reshape(NUM_FRAMES, -1)
    adc_data = np.apply_along_axis(DCA1000.organize, 1, adc_data, num_chirps=numChirpsPerFrame, num_rx=numRxAntennas, num_samples=numADCSamples)
    
    while i < 50:
        print(i)
        # Extract adc_data for the ith frame
        adc_data_frame = adc_data[i]
        adc_data_frame = np.expand_dims(adc_data_frame, axis=0)
        i+=1

        try:
            radar_cube = dsp.range_processing(adc_data_frame[0], window_type_1d=Window.BLACKMAN)
            radar_cube1 = np.expand_dims(radar_cube, axis=1)
            rangechirp = np.concatenate((radar_cube1[0::3],radar_cube1[1::3],radar_cube1[2::3]),axis=1)
            rangechirp = np.abs(rangechirp).sum(axis=(1,2))
            rangefft_out = np.abs(radar_cube).sum(axis=(0,1))
            det_matrix, aoa_input = dsp.doppler_processing(radar_cube, num_tx_antennas=3, clutter_removal_enabled=True, window_type_2d=Window.HAMMING)
            det_matrix_vis = np.fft.fftshift(det_matrix, axes=1)

            rangeResult = np.transpose(np.stack([radar_cube[0::3], radar_cube[1::3], radar_cube[2::3]], axis=1),axes=(1,2,0,3))
            
            # Find the PCD 
            ranges, dopplers, points = get_pcd(det_matrix, aoa_input)
            points = points.T
            pointcloud = np.array([[points[j][0], points[j][1], points[j][2], dopplers[j], ranges[j]] for j in range(len(dopplers))])
            
            # Estimate alpha
            alpha = estimate_alpha(pointcloud)
            
            # Static/dynamic segregation 
            static_pcd, static_range_bins = static_clusters(pointcloud, alpha)

            # Estimate translational speed 
            vel_array_frame = speed_estimation_fn(static_range_bins, rangeResult, static_pcd, alpha)

            # -- Plotting: add data for frame i 
            frames_data.append((rangechirp, det_matrix_vis, vel_array_frame.mean(), pointcloud, i))
        
        except Exception as e:
            print(e)
            continue
    
    ani = FuncAnimation(fig, update, frames=frames_data, blit=False, interval=200)

    # Save as MP4 
    ani.save('radar_animation.mp4', writer='ffmpeg', fps=5)

    plt.close()
