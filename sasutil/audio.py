import os.path
from pathlib import Path

from file import FileFinder
from pydub import AudioSegment


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
        # rate, audio = wav_read(file)
        # audio = audio
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
        segment[:1000 * self.trim].export(exp_path.joinpath(file.name))
        print("Exported", file.name, "to", rel_path)


if __name__ == '__main__':
    formatter = AudioFormatter(os.path.abspath('../resources'))
    formatter.format_all()
