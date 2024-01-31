import os.path
import time
from pathlib import Path

from constants import TAGS
from file import FileFinder, JsonFileIO, CsvFileIO
from pydub import AudioSegment
from queue import Queue

class AudioFormatter:
    finder: FileFinder
    location: Path
    trim: float  # length to trim files at

    def __init__(self, in_dir, trim=10):
        self.location = in_dir
        self.finder = FileFinder(self.location, {'wav', 'mp3', 'ogg'})
        self.trim = trim

    def format_all(self):
        while True:
            path = self.finder.get_next_file()
            if not path:
                break
            elif path is ...:
                continue
            else:
                self._format_file(path)

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
            res.put((curr[0], sub, curr[-1]))

        print("Attempting to encode to JSON")
        self.json_file.add_entries(res)
        print("Encoded to JSON - DONE")


if __name__ == '__main__':
    organizer = AudioOrganizer('../resources/data')
    organizer.init_json()
    # formatter = AudioFormatter(os.path.abspath('../resources'))
    # formatter.format_all()
