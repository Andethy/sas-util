import json
import os
from pathlib import Path
from queue import Queue
import random
from typing import TextIO

from constants import JSON_FIELDS, JSON_ONSET_PATH, JSON_FEATURES_PATH


class FileFinder(object):
    suffixes: set[str]
    _queue: Queue[Path]

    def __init__(self, root, suffixes):
        self.root = Path(root)
        self._queue = Queue()
        self.suffixes = suffixes

        self._queue.put(self.root)

    def get_next_file(self):
        if self._queue.empty():
            return None

        res = self._queue.get()
        # print(res, res.suffix)
        if res.is_dir():
            for child in res.iterdir():
                if child.suffix[1:].lower() in self.suffixes or child.is_dir():
                    print(f'Added {child.name} to queue [{"folder" if child.is_dir() else child.suffix}]')
                    self._queue.put(child)
            return Ellipsis
        else:
            return res if res.suffix[1:] in self.suffixes else None

    def reset(self):
        self._queue = Queue()
        self._queue.put(self.root)

    def get_rel_path(self, file: Path):
        return Path(os.path.relpath(file.parent, self.root))

    def get_full_path(self, file: Path):
        return self.root.joinpath(file)


class BaseFileIO:
    file: TextIO
    file_path: str
    data: list

    def __init__(self, file_path='', file=...):
        self.file = file
        self.file_path = file_path
        self.data = []

        if file_path:
            try:
                self.set_file(self.file_path)
            except IOError:
                pass

    def set_file(self, file_path):
        self.file_path = str(file_path)
        self.file = open(str(self.file_path))

    def manipulate_file(self, data):
        self.file.close()
        self.file = open(self.file_path, 'w')
        self.file.writelines(data)
        self.file.close()

    def get_data(self):
        return self.data

    def get_text(self):
        try:
            return self.file.read().splitlines()
        except ValueError:
            self.file = open(self.file_path, 'r')
            return self.file.read().splitlines()


class CsvFileIO(BaseFileIO):

    def __init__(self, *args):
        super().__init__(*args)
        self.entries = len(self.file.readlines()) - 1
        self.file.seek(0)
        self.headers = self.get_next()

    def get_next(self):
        row = self.file.readline()[:-2].replace('\t', '').split('""')
        row[0] = row[0].replace('"', '')
        return row


class JsonFileIO(BaseFileIO):
    def __init__(self, *args, fields=None):
        self.fields = fields
        super().__init__(*args)

    def add_entries(self, entries, reset=True):

        with open(self.file_path, 'r', encoding="utf-8", errors="replace") as self.file:
            self.data = json.load(self.file)

        if reset:
            self.data = []

        print(entries[0].keys())
        for n in range(len(entries)):
            entry = {}

            items = entries.get() if type(entries) is Queue else entries[n] if type(entries[n]) is list else list(entries[n].values())
            for index, key in enumerate(self.fields if self.fields is not None else entries[0].keys()):
                entry[key] = items[index]
            self.data.append(entry)

        self.file = open(self.file_path, 'w', encoding="utf-8", errors="replace")
        self.file.write(json.dumps(self.data, sort_keys=False, indent=4))
        self.file.close()

    def add_entries_dict(self, data, reset=True):
        with open(self.file_path, 'r', encoding="utf-8", errors="replace") as self.file:
            self.data = json.load(self.file)

        if reset:
            self.data = data
        else:
            for key, val in data.items():
                self.data[key] = val

        self.file = open(self.file_path, 'w', encoding="utf-8", errors="replace")
        self.file.write(json.dumps(self.data, sort_keys=False, indent=4))
        self.file.close()

    def add_fields(self, **kwargs):
        with open(self.file_path, 'r') as self.file:
            self.data = json.load(self.file)
            for coll in self.data:
                for field, default in kwargs.items():
                    coll[field] = default
        with open(self.file_path, 'w') as self.file:
            self.file.write(json.dumps(self.data, sort_keys=False, indent=4))

    def clear_entries(self):
        with open(self.file_path, 'w', encoding="utf-8", errors="replace") as self.file:
            self.file.write(json.dumps([], sort_keys=False, indent=4))

    def rem_fields(self, *args):
        with open(self.file_path, 'r') as self.file:
            self.data = json.load(self.file)
            for coll in self.data:
                for field in args:
                    del coll[field]
        with open(self.file_path, 'w') as self.file:
            self.file.write(json.dumps(self.data, sort_keys=False, indent=4))

    def get_entries(self):
        try:
            with open(self.file_path, 'r') as self.file:
                self.data = json.load(self.file)
                # print("DATA:", self.data[0], "..." if len(self.data) > 1 else "")
        except FileNotFoundError:
            print("NOT FOUND?")
        return self.data

    def numerize_entries(self, **fields):
        with open(self.file_path, 'r') as self.file:
            self.data = json.load(self.file)
            for coll in self.data:
                for field, tp in fields.items():
                    coll[field] = tp(coll[field])
            print(self.data)
        with open(self.file_path, 'w') as self.file:
            self.file.write(json.dumps(self.data, sort_keys=False, indent=4))


def fma():
    js1 = JsonFileIO('../resources/fma/mfcc.json')
    js1.rem_fields("ONSET_BUCKET")
    js2 = JsonFileIO('../resources/fma/onset.json')
    js2.add_fields(ONSET=-1)


def test():
    js = JsonFileIO('../resources/fma/buckets.json', fields=((0, 0), (0, 1), (1, 1)))
    js.add_entries_dict(
        {
            "(0, 0)": ["000002.mp3", "000003.mp3"],
            "(0, 1)": ["000002.mp3", "000003.mp3"],
            "(0, 2)": ["000002.mp3", "000003.mp3"],
            "(1, 0)": ["000002.mp3", "000003.mp3"],
            "(1, 1)": ["000002.mp3", "000003.mp3"],
            "(1, 2)": ["000002.mp3", "000003.mp3"],
            "(2, 0)": ["000002.mp3", "000003.mp3"],
            "(2, 1)": ["000002.mp3", "000003.mp3"],
            "(2, 2)": ["000002.mp3", "000003.mp3"],
        })


def trim_folders(folder_path, keep=20):
    random.seed(42693)
    # Set the path to the main folder
    main_folder_path = Path(folder_path)

    # Iterate over each sub_folder in the main folder
    for sub_folder in os.listdir(main_folder_path):
        sub_folder_path = os.path.join(main_folder_path, sub_folder)

        # Ensure the path is indeed a directory to avoid processing any files in the main folder
        if os.path.isdir(sub_folder_path):
            # List all files in the sub_folder
            files = [f for f in os.listdir(sub_folder_path) if os.path.isfile(os.path.join(sub_folder_path, f))]

            # Continue only if there are more than 20 files to delete
            if len(files) > keep:
                # Randomly select 20 files to keep
                files_to_keep = random.sample(files, keep)

                # Delete all files not in the 'files_to_keep' list
                for f in files:
                    if f not in files_to_keep:
                        file_path = os.path.join(sub_folder_path, f)
                        print(f'Attempting to remove {file_path}')
                        os.remove(file_path)  # Use shutil.rmtree(file_path) if directories are also to be considered
        print("All files removed that could be removed")


if __name__ == '__main__':
    trim_folders('/Volumes/Music/Robotics/fma_buckets', 100)
    # test()
    # fma()
    # csv = CsvFileIO('../resources/tagatune/annotations.csv')
    # print(str(csv.headers).replace(',', ',\n'))

    # js = JsonFileIO(JSON_FEATURES_PATH)
    # js.numerize_entries(Index=int,
    #                     mfcc_mean=float,
    #                     mfcc_min=float,
    #                     mfcc_max=float,
    #                     melSpec_mean=float,
    #                     melSpec_min=float,
    #                     melSpec_max=float,
    #                     chromaVec_mean=float,
    #                     chromaVec_min=float,
    #                     chromaVec_max=float,
    #                     roll_mean=float,
    #                     roll_min=float,
    #                     roll_max=float,
    #                     zcr_mean=float,
    #                     zcr_min=float,
    #                     zcr_max=float)
    # js.numerize_entries(Index=int, Min=float, Max=float, Energy=float)
    # js.rem_fields('MFCCS_Bucket', 'ONSET_BUCKET')

