def load_audio_features(data_path):
    """
    Load audio files from subfolders (GTZAN format).
    Returns:
        X: feature matrix
        y: genre labels
        file_names
    """
    features = []
    labels = []
    file_names = []

    for genre in os.listdir(data_path):
        genre_path = os.path.join(data_path, genre)

        if os.path.isdir(genre_path):
            for file in os.listdir(genre_path):
                if file.endswith(".wav"):
                    file_path = os.path.join(genre_path, file)

                    mfcc = extract_mfcc(file_path)

                    if mfcc is not None:
                        features.append(mfcc)
                        labels.append(genre)
                        file_names.append(file)

    X = np.array(features)
    y = np.array(labels)

    return X, y, file_names
