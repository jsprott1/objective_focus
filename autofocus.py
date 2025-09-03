import numpy as np
from camera_control import get_frame
from warnings import warn
from enum import Enum

def focus(camera, stage, scan_range, factor, max_iter, pos_bounds, error_on_no_trend=True, peak_in_range=False):
    frames = []
    positions = []
    initial_pos = stage.get_motor_position("o")
    stage.move_to("o", initial_pos - scan_range/2)
    frames.append(get_frame(camera))
    positions.append(stage.get_motor_position("o"))
    for i in range(int(2/factor)):
        stage.move_to("o", stage.get_motor_position("o") + 1/2*scan_range*factor)
        frames.append(get_frame(camera))
        positions.append(stage.get_motor_position("o"))
    contrast = contrast_metric(frames)
    peak = peak_type(contrast)
    if 0 == max_iter:
        if peak_in_range or peak == Peaks.CENTRE_PEAK or not error_on_no_trend:
            stage.move_to("o", positions[np.argmax(contrast)])
            return np.max(contrast)
        else:
            warn("Focus not found")
            stage.move_to("o", positions[np.argmax(contrast)])
            return np.max(contrast)
    elif peak_in_range or peak == Peaks.CENTRE_PEAK:
        stage.move_to("o", positions[np.argmax(contrast)])
        return focus(camera, stage, scan_range * factor, factor, max_iter - 1, pos_bounds - (stage.get_motor_position("o") - initial_position), error_on_no_trend = error_on_no_trend, peak_in_range=True)
    elif peak == Peaks.NO_PEAK: 
        if error_on_no_trend:
            warn("Focus not found")
            stage.move_to("o", positions[np.argmax(contrast)])
            return np.max(contrast)
        else:
            stage.move_to("o", initial_pos)
            return focus(camera, stage, scan_range / factor, factor, max_iter - 1, pos_bounds, error_on_no_trend = error_on_no_trend, peak_in_range = False)
    elif peak == Peaks.EDGE_PEAK:
        stage.move_to("o", positions[np.argmax(contrast)])
        return focus(camera, stage, scan_range, factor, max_iter - 1, pos_bounds - (stage.get_motor_position("o") - initial_position), error_on_no_trend = error_on_no_trend, peak_in_range = False)
    
# frames is a 3d numpy array of frames (gets mapped to [0,1]. Returns mean square difference between each pixel and its neighbourhood for each frame.
def contrast_metric(frames, neighbourhood=1):
    frames /= np.max(frames)
    # Pad the edges so that they don't affect each other during roll
    frames_padded = np.pad(frames, ((0,0),(neighbourhood,neighbourhood),(neighbourhood,neighbourhood)), mode='edge')
    shift_vals = np.linspace(-neighbourhood, neighbourhood+1)
    sum_ = np.zeros_like(frames_padded)
    for x in shift_vals:
        for y in shift_vals:
            sum_ = sum_ + np.roll(frames_padded, (0,x,y))
    average = (sum_/np.size(shift_vals)**2)[:, neighbourhood:-neighbourhood,neighbourhood:-neighbourhood] 
    diff2 = (frames - average)**2
    return np.mean(diff2, (1,2))

def peak_type(contrast, threshold=0.2):
    max_, max_i = np.max(contrast), np.argmax(contrast)
    if max_ - np.mean(contrast) < threshold:
        return Peaks.NO_PEAK
    if max_i == 0 or max_i == contrast.shape[0] - 1: 
        return Peaks.EDGE_PEAK
    else:
        return Peaks.CENTRE_PEAK

class Peaks(Enum):
    NO_PEAK = 1
    EDGE_PEAK = 2
    CENTRE_PEAK = 3
