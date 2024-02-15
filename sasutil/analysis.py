import collections
import math
from queue import Queue

from constants import *
from file import JsonFileIO


# librosa.feature.mfcc
# librosa.onset.onset_detect

class Analyzer:
    def __init__(self, in_dir, fields):
        self.fields = fields
        self.out_file = JsonFileIO(in_dir, fields=fields)

    def analyze_data(self, in_field, out_field, buckets: int = 4):
        data = self.out_file.get_entries()
        buffer = []
        max_res = 0
        min_res = math.inf
        print(in_field)
        print(self.fields)
        in_loc = self.fields.index(in_field)
        out_loc = self.fields.index(out_field)

        # for dt in data:
        #     temp = dt[in_field]
        #     if temp > max_res:
        #         max_res = dt[in_field]
        #     elif temp < min_res:
        #         min_res = dt[in_field]

        for dt in data:
            dt[out_field] = self.calculate_bucket(buckets, min_res, max_res, dt[in_field])
            buffer.append(list(dt[field] for field in self.fields))
            # print(*tuple(dt[field] for field in self.fields))
        buffer = sorted(buffer, key=lambda x: x[in_loc])
        for n in range(len(buffer)):
            buffer[n][out_loc] = min(int(n // (len(buffer) // buckets)), buckets - 1)
        self.out_file.add_entries(buffer)

    def classify_data(self, buckets, *jsons):
        data = collections.defaultdict(list)
        temp = collections.defaultdict(str)
        for bucket in buckets:
            data[bucket] = []

        for json in jsons:
            file = JsonFileIO(json)
            for curr in file.get_entries():
                temp[curr["Name"]] = temp[curr["Name"]] + str(curr["BUCKET"])

        for file, key in temp.items():
            data[key].append(file)

        self.out_file.add_entries_dict(data)

    @staticmethod
    def calculate_bucket(buckets: int, lb: float, ub: float, ac: float):
        bucket = 0
        curr = lb
        for n in range(0, buckets):
            curr += (ub - lb) / buckets
            if ac <= curr:
                break
            bucket += 1
        return bucket


if __name__ == '__main__':
    # analyzer = Analyzer(JSON_ONSET_PATH, fields=ONSET_FIELDS)
    # analyzer.analyze_data('Onsets', 'BUCKET')
    analyzer = Analyzer(JSON_BUCKETS_PATH, fields=OUTPUT_BUCKETS)
    analyzer.classify_data(OUTPUT_BUCKETS, JSON_MFCC_PATH, JSON_ONSET_PATH)

