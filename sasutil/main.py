from analysis import AudioAnalyzer
from audio import AudioOrganizer
from constants import *
from file import JsonFileIO, trim_folders


def clusterize_fma(out_name: str, buckets: int = 10):
    analyzer = AudioAnalyzer(FMA_BUCKETS_PATH, fields=OUTPUT_BUCKETS)
    analyzer.correlate_buckets_old(JsonFileIO(FMA_FEATURES_PATH), list(FEATURES_FIELDS_SMALL), buckets)
    analyzer.classify_data(OUTPUT_BUCKETS, FMA_FEATURES_PATH)
    # organizer = AudioOrganizer('/Volumes/Music/Robotics/fma', json_file=JSON_BUCKETS_PATH)
    # organizer.organize_files_fma(out_name)
    # trim_folders(f'/Volumes/Music/Robotics/{out_name}', 100)


def clusterize_gtzan(buckets: int = 10):
    analyzer = AudioAnalyzer(GTZ_BUCKETS_PATH, fields=OUTPUT_BUCKETS)
    analyzer.correlate_buckets(JsonFileIO(GTZ_FEATURES_PATH), list(FEATURES_FIELDS_SMALL), buckets)
    analyzer.classify_data(OUTPUT_BUCKETS, GTZ_FEATURES_PATH)
    organizer = AudioOrganizer('/Volumes/Music/Robotics/gtzan_formatted', json_file=GTZ_BUCKETS_PATH)
    organizer.organize_files_buckets('/Volumes/Music/Robotics/gtzan_onsets')


if __name__ == "__main__":
    clusterize_gtzan(10)
