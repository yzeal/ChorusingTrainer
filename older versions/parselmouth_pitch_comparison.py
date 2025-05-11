
import parselmouth
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def extract_smoothed_pitch(path, f0_min=75, f0_max=500, smoothing=True):
    snd = parselmouth.Sound(path)
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=f0_min, pitch_ceiling=f0_max)
    
    # Extract pitch values and confidence
    pitch_values = pitch.selected_array['frequency']
    confidence = pitch.selected_array['strength']  # 0.0 to 1.0
    times = pitch.xs()

    # Filter: Only keep high-confidence voiced frames
    threshold = 0.6
    voiced_indices = np.where((confidence > threshold) & (pitch_values > 0))[0]
    
    if len(voiced_indices) < 2:
        print("Too few voiced frames detected.")
        return times, pitch_values  # return raw data anyway

    # Thinning: take every N-th voiced frame
    step = max(1, len(voiced_indices) // 10)
    selected = voiced_indices[::step]
    x_sparse = times[selected]
    y_sparse = pitch_values[selected]

    if smoothing and len(x_sparse) >= 4:
        f_interp = interp1d(x_sparse, y_sparse, kind='cubic', fill_value="extrapolate")
        x_dense = np.linspace(x_sparse[0], x_sparse[-1], 300)
        y_dense = f_interp(x_dense)
        return x_dense, y_dense
    else:
        return x_sparse, y_sparse

def plot_two_pitch_curves(path_native, path_user):
    x1, y1 = extract_smoothed_pitch(path_native)
    x2, y2 = extract_smoothed_pitch(path_user)

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axs[0].plot(x1, y1, label="Native", linewidth=2)
    axs[0].set_title("Native Speaker (Smoothed Pitch)")
    axs[0].set_ylabel("Hz")
    axs[0].legend()

    axs[1].plot(x2, y2, label="Your Recording", color="orange", linewidth=2)
    axs[1].set_title("Your Pitch (Smoothed)")
    axs[1].set_ylabel("Hz")
    axs[1].legend()

    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

# Example usage:
# plot_two_pitch_curves("native.wav", "user.wav")
