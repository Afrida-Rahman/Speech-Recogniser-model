import librosa
import os
import json

dataset_path = 'audio'
json_path = 'data.json'
samples_to_consider = 22050


def prepare_dataset(dataset_path, json_path, n_mfcc=13, hop_length=512, n_fft=2048):
    data = {
        "mappings": [],
        "labels": [],
        "MFCCs": [],
        "files": []
    }

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path:
            category = dirpath.split('/')[-1]
            data['mappings'].append(category)

            print(f"Processing {category}")
            for f in filenames:
                filepath = os.path.join(dirpath, f)
                signal, sr = librosa.load(filepath)
                if len(signal) >= samples_to_consider:
                    signal = signal[:samples_to_consider]
                    MFCCs = librosa.feature.mfcc(signal, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)

                    data["labels"].append(i - 1)
                    data["MFCCs"].append(MFCCs.T.tolist())
                    data["files"].append(filepath)
                    print(f"{filepath} : {i - 1}")

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


prepare_dataset(dataset_path, json_path)
