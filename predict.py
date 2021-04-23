import librosa
import tensorflow as tf
import numpy as np

SAVED_MODEL_PATH = "model.h5"
SAMPLES_TO_CONSIDER = 22050


class keyword:
    model = None
    _mapping = [
        "down",
        "off",
        "on",
        "no",
        "yes",
        "stop",
        "up",
        "right",
        "left",
        "go"
    ]
    _instance = None

    def predict(self, file_path):
        MFCCs = self.preprocess(file_path)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        predictions = self.model.predict(MFCCs)
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mapping[predicted_index]
        return predicted_keyword

    def preprocess(self, file_path, num_mfcc=13, n_fft=2048, hop_length=512):
        signal, sample_rate = librosa.load(file_path)

        if len(signal) >= SAMPLES_TO_CONSIDER:
            signal = signal[:SAMPLES_TO_CONSIDER]

            MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        return MFCCs.T


def Keyword_Predict():
    if keyword._instance is None:
        keyword._instance = keyword()
        keyword.model = tf.keras.models.load_model(SAVED_MODEL_PATH)
    return keyword._instance


if __name__ == "__main__":
    kss = Keyword_Predict()
    kss1 = Keyword_Predict()

    assert kss is kss1

    keyword = kss.predict("test/no.wav")
    print(f"Predicted class : {keyword}" )
