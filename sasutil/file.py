import json
import os
from pathlib import Path
from queue import Queue
from typing import TextIO

from constants import JSON_FIELDS


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
    def __init__(self, *args):
        super().__init__(*args)

    def add_entries(self, entries: Queue):

        with open(self.file_path, 'r', encoding="utf-8", errors="replace") as self.file:
            self.data = json.load(self.file)

        for n in range(entries.qsize()):
            entry = {}
            items = entries.get()
            for index, key in enumerate(JSON_FIELDS):
                entry[key] = items[index]
            self.data.append(entry)

        self.file = open(self.file_path, 'w', encoding="utf-8", errors="replace")
        self.file.write(json.dumps(self.data, sort_keys=False, indent=4))
        self.file.close()

    def clear_entries(self):
        with open(self.file_path, 'w', encoding="utf-8", errors="replace") as self.file:
            self.file.write(json.dumps([], sort_keys=False, indent=4))

    def get_entries(self):
        try:
            with open(self.file_path, 'r') as self.file:
                self.data = json.load(self.file)
                print("DATA:", self.data)
        except FileNotFoundError:
            print("NOT FOUND?")
        print(self.data)
        return self.data


if __name__ == '__main__':
    csv = CsvFileIO('../resources/tagatune/annotations.csv')
    print(str(csv.headers).replace(',', ',\n'))
