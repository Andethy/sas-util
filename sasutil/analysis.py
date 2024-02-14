import math
from queue import Queue

from constants import JSON_MFCC_PATH, ENERGY_FIELD, MFCC_FIELD, MFCC_FIELDS
from file import JsonFileIO


# librosa.feature.mfcc
# librosa.onset.onset_detect

class Analyzer:
    def __init__(self, in_dir, fields):
        self.fields = fields
        self.out_file = JsonFileIO(in_dir, fields=fields)

    def analyze_data(self, in_field: tuple, out_field: tuple, buckets: int = 4):
        data = self.out_file.get_entries()
        buffer = []
        max_res = 0
        min_res = math.inf
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
    analyzer = Analyzer(JSON_MFCC_PATH, MFCC_FIELDS)
    analyzer.analyze_data(ENERGY_FIELD, MFCC_FIELD)
