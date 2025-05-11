
def extract_smoothed_pitch(self, path, f0_min=75, f0_max=500):
    snd = parselmouth.Sound(path)
    duration = snd.get_total_duration()
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=f0_min, pitch_ceiling=f0_max)
    pitch_values = pitch.selected_array['frequency']
    confidence = pitch.selected_array['strength']
    times = pitch.xs()

    # Use voiced and confident pitch values
    voiced_indices = np.where((confidence > 0.6) & (pitch_values > 0))[0]
    if len(voiced_indices) < 4:
        return times, pitch_values

    # Use 10 points per second of audio
    n_points = max(4, int(duration * 10))
    step = max(1, len(voiced_indices) // n_points)
    selected = voiced_indices[::step]

    x_sparse = times[selected]
    y_sparse = pitch_values[selected]

    if len(x_sparse) >= 4:
        f_interp = interp1d(x_sparse, y_sparse, kind='cubic', fill_value="extrapolate")
        x_dense = np.linspace(x_sparse[0], x_sparse[-1], 300)
        y_dense = f_interp(x_dense)
        return x_dense, y_dense
    else:
        return x_sparse, y_sparse
