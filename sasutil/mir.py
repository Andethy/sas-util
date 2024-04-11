from pathlib import Path

import numpy as np
from pydub import AudioSegment

from constants import SAMPLE_RATE, void
from file import JsonFileIO

from librosa import load, feature, onset, frames_to_time


class FeatureExtractor:
    def __init__(self, in_path: str, out_path: str, global_rate=SAMPLE_RATE):
        self.files = Path(in_path)
        self.out = JsonFileIO(out_path)
        self.global_rate = global_rate

    def analyze_all_tracks(self):
        data: dict = self.out.get_entries()
        for file in self.files.iterdir():
            if file.suffix != '.mp3':
                continue

            # Get track array
            track = file.name[:-4]
            temp_data = data[track] if track in data else {}
            track_arr = self.get_normalized_mp3_array(file)
            print(f'Extracting info for track ID {track}')

            # Extract features
            temp_data["Tempo"] = float(self.get_tempo(track_arr))
            temp_data["Onsets"] = self.get_onsets(track_arr).tolist()
            temp_data["MFCC"] = self.get_mfcc(track_arr).tolist()
            temp_data["Centroid"] = self.get_centroid(track_arr).tolist()
            temp_data["Bandwidth"] = self.get_bandwidth(track_arr).tolist()
            temp_data["Contrast"] = self.get_contrast(track_arr).tolist()
            temp_data["Flatness"] = self.get_flatness(track_arr).tolist()
            temp_data["Rolloff"] = self.get_rolloff(track_arr).tolist()

            # Store modified data
            data[track] = temp_data
        self.out.add_entries_dict(data)
        print("Entries added back to JSON - DONE")

    def get_normalized_mp3_array(self,
                                 file: Path) -> np.ndarray:  # audio_arr = audio_arr.reshape((-1, 2)) if audio_mp3.channels > 1 else audio_arr
        """


        :param file: path to mp3 file
        :return: a numpy array representing the mp3 file
        """
        # audio_mp3 = AudioSegment.from_file(file)  # .split_to_mono() ?
        # audio_arr = np.array(audio_mp3.get_array_of_samples(), dtype=np.float64)
        # audio_arr /= np.max(np.abs(audio_arr))
        audio_arr = load(file, sr=self.global_rate)[0]
        audio_arr /= np.max(np.abs(audio_arr))
        return audio_arr

    def get_tempo(self, audio: np.ndarray) -> np.float64:
        res = feature.tempo(y=audio, sr=self.global_rate)
        # noinspection PyTypeChecker
        return res[0]

    def get_onsets(self, audio: np.ndarray) -> np.ndarray:
        res = onset.onset_detect(y=audio, sr=self.global_rate)
        return frames_to_time(res)

    def get_centroid(self, audio: np.ndarray) -> np.ndarray:
        res = feature.spectral_centroid(y=audio, sr=self.global_rate)
        return res[0]

    def get_bandwidth(self, audio: np.ndarray) -> np.ndarray:
        res = feature.spectral_bandwidth(y=audio, sr=self.global_rate)
        return res[0]

    def get_contrast(self, audio: np.ndarray) -> np.ndarray:
        res = feature.spectral_contrast(y=audio, sr=self.global_rate)
        return res[0]

    def get_flatness(self, audio: np.ndarray) -> np.ndarray:
        void(self.global_rate)
        res = feature.spectral_flatness(y=audio)
        return res[0]

    def get_rolloff(self, audio: np.ndarray) -> np.ndarray:
        res = feature.spectral_rolloff(y=audio, sr=self.global_rate)
        return res[0]

    def get_mfcc(self, audio: np.ndarray) -> np.ndarray:
        res = feature.mfcc(y=audio, sr=self.global_rate)
        return res


if __name__ == '__main__':
    extractor = FeatureExtractor('../resources/study/dataset', '../resources/analysis/features.json')
    extractor.analyze_all_tracks()
