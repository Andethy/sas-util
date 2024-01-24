import os
from pathlib import Path
from queue import Queue


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
