from process_data import *
from scipy.fft import rfft, rfftfreq



def create_dataset(root_dir: str, n_s: int, model_type: str, fs: int = 256, 
                   Tstart: float = 1.5, Tend: float = 3.5, encoding : str = "rate") -> tuple:
    """
    Prepares datasets for different model types based on raw EEG data.

    Parameters:
    - root_dir (str): The root directory containing the data.
    - n_s (int): The subject number.
    - model_type (str): The target model type ("SNN", "ANN", "SVM").
    - fs (int): Sampling frequency, default is 256 Hz.
    - Tstart (float): Start time in seconds for trimming the signal.
    - Tend (float): End time in seconds for trimming the signal.
    - encoding (str): Type of encoding used for spikes
    Returns:
    - tuple: Processed dataset (X, Y) ready for the specified model type.
    """
    datatype = "eeg"
    X, Y = extract_data_from_subject(root_dir, n_s, datatype)



    X_trimmed = select_time_window(X=X, t_start=Tstart, t_end=Tend, fs=fs)
    # Define conditions and classes to use for transformation

    # Borde egentligen ha conditions + classes som parameter 
    Conditions = [["Inner"]]
    Classes = [["all"]]

    # Apply classification transformation to filter data by conditions and classes
    X, Y = transform_for_classificator(X_trimmed, Y, classes=Classes, conditions=Conditions)

   
    X = X * (10**6)  # Convert from volts to microvolts
    

    
    # Process data depending on model type
    if model_type.upper() == "SNN":
        # Spike encoding
        X_processed = convert_to_spikes(X, fs, encoding)
    
    elif model_type.upper() == "ANN":
        
        X_processed = normalize_data(X) 
    
    elif model_type.upper() == "SVM":
        # Compute FFT and extract spectral features
        X_processed = extract_fft_features(X, fs)
    
    else:
        raise ValueError("Unsupported model type. Choose 'SNN', 'ANN', or 'SVM'.")
    
    return X_processed, Y[:, 1]


def create_dataset_for_all_subjects(root_dir: str, model_type: str, fs: int = 256, 
                                    Tstart: float = 1.5, Tend: float = 3.5, encoding: str = "rate") -> tuple:
    """
    Creates a dataset for all subjects by calling the create_dataset function for each subject.

    Parameters:
    - root_dir (str): The root directory containing the data.
    - model_type (str): The target model type ("SNN", "ANN", "SVM").
    - fs (int): Sampling frequency, default is 256 Hz.
    - Tstart (float): Start time in seconds for trimming the signal.
    - Tend (float): End time in seconds for trimming the signal.
    - encoding (str): Type of encoding used for spikes.
    - num_subjects (int): Total number of subjects to process.

    Returns:
    - tuple: Processed dataset (X, Y) for all subjects combined.
    """
    X_all = []
    Y_all = []

    for i in range(1, 11):
        X, Y = create_dataset(root_dir, i, model_type, fs, Tstart, Tend, encoding)
        X_all.append(X)
        Y_all.append(Y)

    # Concatenate data for all subjects along the first axis (samples)
    X_all = np.concatenate(X_all, axis=0)
    Y_all = np.concatenate(Y_all, axis=0)

    return X_all, Y_all


def convert_to_spikes(X: np.ndarray, fs: int, encoding: str) -> np.ndarray:
    """
    Convert EEG signals into spike trains for SNNs.

    Parameters:
    - X (np.ndarray): Raw EEG data of shape (samples, channels, time).
    - fs (int): Sampling frequency.
    - encoding (str): Encoding type ("RATE", "TEMPORAL", or "POISSON").

    Returns:
    - np.ndarray: Spike-encoded data.
    """
    # Normalize each channel (axis 1 = channel dimension)
    X_normalized = (X - X.min(axis=1, keepdims=True)) / (X.max(axis=1, keepdims=True) - X.min(axis=1, keepdims=True))


    if encoding.upper() == "RATE":
        # Rate encoding: convert amplitude to spike train where signal exceeds threshold
        #threshold = 1/20  # Adjustable threshold
        #spike_trains = (X_normalized > threshold).astype(np.float32)
        #return spike_trains
        #print("Hello")
        # Temporal encoding: signal amplitude converted to spike timing

        duration = 100
        max_rate = 100

        samples, channels, time_steps = X.shape
        spike_trains = np.zeros((samples, channels, duration))
        
        for sample_idx in range(samples):
            for channel_idx in range(channels):
                time_series = X_normalized[sample_idx, channel_idx, :]
                for t in range(duration):
                    value = time_series[t % time_steps]  # Wrap time data if duration > time_steps
                    firing_rate = value * max_rate  # Convert value to firing rate
                    spike_prob = firing_rate / 1000  # Convert rate to spike probability per ms
                    spike_trains[sample_idx, channel_idx, t] = np.random.rand() < spike_prob
        
        return spike_trains
        
        
        
        
        '''
        n_time_steps = X.shape[2]
        n_neurons = len(X)
        max_rate = 100

        spike_trains = np.zeros((n_time_steps, n_neurons))

        for neuron_idx, value in enumerate(X_normalized):
            # Firing probability proportional to value
            firing_rate = value * max_rate  # Convert value to firing rate
            spike_prob = firing_rate / 1000  # Convert rate to spike probability per ms
            
            # Generate spikes for this neuron over time
            spike_trains[:, neuron_idx] = np.random.rand(n_time_steps) < spike_prob
        
        return spike_trains
        '''








        max_time = 100  # Maximum spike time (e.g., 100 ms)
        spike_times = (X_normalized * max_time).astype(np.int32)

        # Initialize the spike train array
        spike_trains = np.zeros((X.shape[0], X.shape[1], max_time), dtype=np.float32)
        
        # Create spikes at the encoded time steps
        for sample in range(X.shape[0]):
            for channel in range(X.shape[1]):
                for t in range(X.shape[2]):
                    time = spike_times[sample, channel, t]
                    if time < max_time:
                        spike_trains[sample, channel, time] = 1  # Set spike at the specific time index
        return spike_trains


    elif encoding.upper() == "TEMPORAL":
        # Temporal encoding: signal amplitude converted to spike timing
        max_time = 100  # Maximum spike time (e.g., 100 ms)
        spike_times = (X_normalized * max_time).astype(np.int32)

        # Initialize the spike train array
        spike_trains = np.zeros((X.shape[0], X.shape[1], max_time), dtype=np.float32)
        
        # Create spikes at the encoded time steps
        for sample in range(X.shape[0]):
            for channel in range(X.shape[1]):
                for t in range(X.shape[2]):
                    time = spike_times[sample, channel, t]
                    if time < max_time:
                        spike_trains[sample, channel, time] = 1  # Set spike at the specific time index
        return spike_trains

    elif encoding.upper() == "POISSON":
        # Poisson encoding: generate spikes based on signal amplitude
        spike_trains = np.random.poisson(lam=X_normalized)

        # Ensure the result is in float32 type
        spike_trains = spike_trains.astype(np.float32)
        return spike_trains

    else:
        raise ValueError("Unsupported encoding type. Use 'RATE', 'TEMPORAL', or 'POISSON'.")



def normalize_data(X: np.ndarray) -> np.ndarray:
    """
    Normalize EEG data for ANN models.
    """
    return (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True)

def extract_fft_features(X: np.ndarray, fs: int) -> np.ndarray:
    """
    Extract FFT features for SVM training.

    Parameters:
    - X (np.ndarray): Raw EEG data of shape (samples, channels, time).
    - fs (int): Sampling frequency.

    Returns:
    - np.ndarray: Processed FFT features.
    """
    N = X.shape[2]  # Time points per sample
    samples_fft = rfft(X, axis=2)
    samples_freq = rfftfreq(N, 1/fs)
    
    # Average across channels and calculate magnitude
    avg_fft = np.mean(samples_fft, axis=1)
    mag_fft = np.sqrt((avg_fft.real ** 2) + (avg_fft.imag ** 2))
    db_fft = 20 * np.log10(mag_fft)
    
    # Frequency range selection (e.g., 4 to 40 Hz)
    freq_mask = (samples_freq >= 4) & (samples_freq <= 40)
    return np.real(db_fft[:, freq_mask])


