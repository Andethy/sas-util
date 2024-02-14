import os.path
import sys
import time
from pathlib import Path

from pydub.exceptions import CouldntDecodeError

from constants import TAGS, REQUIRED, TAG_MAP
from file import FileFinder, JsonFileIO, CsvFileIO
from pydub import AudioSegment
from queue import Queue
from shutil import copyfile, rmtree


class AudioFormatter:
    finder: FileFinder
    location: Path
    trim: float  # length to trim files at

    def __init__(self, in_dir, trim=10):
        self.location = in_dir
        self.finder = FileFinder(self.location, {'wav', 'mp3', 'ogg'})
        self.trim = trim

    def format_all(self):
        print("Attempting to format all files...")
        while True:
            path = self.finder.get_next_file()
            if not path:
                break
            elif path is ...:
                continue
            else:
                try:
                    self._format_file(path)
                except CouldntDecodeError:
                    print(f"ERROR - Could not trim {path.name} - removing it")
                    os.remove(path)
        print("Files formatted that could be formatted - DONE")

    def _format_file(self, file: Path):
        segment = ...
        if file.suffix == '.wav':
            segment = AudioSegment.from_wav(file)
        elif file.suffix == '.mp3':
            segment = AudioSegment.from_mp3(file)
        elif file.suffix == '.ogg':
            segment = AudioSegment.from_ogg(file)

        rel_path = self.finder.get_rel_path(file)
        exp_path = Path(os.path.abspath('../out/' + str(rel_path)))
        exp_path.mkdir(parents=True, exist_ok=True)
        segment[:int(1000 * self.trim)].export(exp_path.joinpath(file.name))
        print("Exported", file.name, "to", rel_path)


class AudioOrganizer:

    def __init__(self, in_dir: str):
        self.finder = FileFinder(in_dir, {'wav', 'mp3', 'ogg'})
        self.csv_file = CsvFileIO('../resources/tagatune/annotations.csv')
        self.json_file = JsonFileIO('../resources/tagatune/annotations.json')

    def init_json(self):
        res = Queue()
        print(f"Attempting to add tracks to buffer")
        while res.qsize() < self.csv_file.entries:
            curr = self.csv_file.get_next()

            if not curr:
                break
            sub = []
            for tag, value in zip(TAGS[1:-1], curr[1:-1]):
                if value == '1':
                    sub.append(tag)
            res.put((int(curr[0]), sub, curr[-1]))

        print("Attempting to encode to JSON")
        self.json_file.add_entries(res)
        print("Encoded to JSON - DONE")

    def organize_files_tgt(self):
        print("Attempting to retrieve data")
        data = self.json_file.get_data()
        print("Attempting to copy files to appropriate directories")
        for entry in data:
            val, tags, path = entry.values()
            skip = True
            # print(val, tags)
            for token in REQUIRED:
                if token in tags:
                    skip = False

            if skip:
                continue

            for tag, rel in TAG_MAP.items():
                if tag in tags:
                    src_path = Path(f'../resources/data/{path}')
                    des_path = Path(f'../out/{rel}/{path[2:]}')
                    Path(f'../out/{rel}').mkdir(parents=True, exist_ok=True)
                    try:
                        copyfile(src_path, des_path)
                    except IsADirectoryError:
                        continue

        # exp_path = Path(os.path.abspath('../out/' + str(rel_path)))
        # exp_path.mkdir(parents=True, exist_ok=True)
        print("Files transferred - DONE")

    def organize_files_fma(self):
        pass


if __name__ == '__main__':
    try:
        # os.remove(Path("../out").absolute())
        rmtree(Path("../out").absolute())
        print("REMOVED OUT")
    except FileNotFoundError:
        print("NOT FOUND")
        pass
    organizer = AudioOrganizer('../resources/data')
    organizer.init_json()
    organizer.organize_files()
    formatter = AudioFormatter(os.path.abspath('../out'), 7)
    formatter.format_all()
