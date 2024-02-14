from collections import deque


# librosa.feature.mfcc
# librosa.onset.onset_detect

class Result:
    max_res = 0
    min_res = 0
    buckets = 5

    def __init__(self, *args, **kwargs):
        self.avg = ...
        if self.avg > Result.max_res:
            Result.max_res = self.avg
        elif self.avg < Result.min_res:
            Result.min_res = self.avg

    def calculate_bucket(self):
        bucket = 0
        curr = Result.min_res
        for n in range(0, Result.buckets):
            curr += (Result.max_res - Result.min_res) / Result.buckets
            if self.avg <= curr:
                break
            bucket += 1
        return bucket


buffer = deque()
