import time
from threading import Lock
from typing import Dict, List, Tuple

from attr import attrib, attrs


@attrs
class SessionDataCache:
    size: int = attrib()
    timed_cache: Dict[str, Tuple[float, Dict]] = attrib(factory=dict)
    lock: Lock = Lock()

    def __getitem__(self, key: str) -> Dict:
        value = None
        self.lock.acquire()
        if key in self.timed_cache:
            last_access_time, value = self.timed_cache[key]
            current_time = time.time()
            self.timed_cache[key] = (current_time, value)  # update last access time
        self.lock.release()

        if value is None:
            raise KeyError

        return value

    def __contains__(self, key: str) -> bool:
        self.lock.acquire()
        present: bool = key in self.timed_cache
        self.lock.release()
        return present

    def __setitem__(self, key: str, value: Dict) -> None:
        current_time = time.time()

        self.lock.acquire()
        self.timed_cache[key] = (current_time, value)
        self.lock.release()

        if len(self.timed_cache) > self.size:
            self._evict_old_entries()

    def _evict_old_entries(self):
        self.lock.acquire()
        # noinspection PyTypeChecker
        time_sorted_dict: List[Tuple[str, Tuple[float, Dict]]] = sorted(
            list(self.timed_cache.items()), key=lambda entry: entry[1][0]
        )
        self.timed_cache = dict(time_sorted_dict[-self.size :])
        self.lock.release()

    def __delitem__(self, key: str):
        if not isinstance(key, str):
            raise NotImplementedError("")

        self.lock.acquire()
        del self.timed_cache[key]
        self.lock.release()
