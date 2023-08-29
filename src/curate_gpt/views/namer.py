from dataclasses import dataclass
from typing import List, Iterable, Iterator, Tuple, Any

from curate_gpt import DBAdapter


@dataclass
class Namer:
    """
    Maps between opaque IDs and names

    TODO: use caching
    """
    stores: List[DBAdapter]

    def map_object(self, obj: Any, paths: List[List[str]], forward=True) -> Any:
        """
        Maps an object to a new object with names instead of IDs
        """
        if isinstance(obj, dict):
            mapped = {}
            for k, v in obj.items():
                if [k] in paths:
                    mapped[k] = self.map_reference(v, forward=forward)
                else:
                    new_paths = [path[1:] for path in paths if path[0] == k]
                    mapped[k] = self.map_object(v, new_paths, forward=forward)
        elif isinstance(obj, list):
            mapped = []
            for v in obj:
                mapped.append(self.map_object(v, paths, forward=forward))
        else:
            mapped = obj
        return mapped

    def map_reference(self, ref: str, forward=True) -> str:
        """
        Maps a reference to a name
        """
        if forward:
            return list(self.ids_to_names([ref]))[0][1]
        else:
            return list(self.names_to_ids([ref]))[0][1]

    def ids_to_names(self, ids: Iterable[str]) -> Iterator[Tuple[str, str]]:
        """
        fetches mappings
        """
        yield from self._query_mappings(ids, self.store.identifier_field(), self.store.label_field())

    def names_to_ids(self, names: Iterable[str]) -> Iterator[Tuple[str, str]]:
        """
        fetches mappings
        """
        yield from self._query_mappings(names, self.store.label_field(), self.store.identifier_field())

    def _query_mappings(self, from_vals: List[str], from_field: str, to_field: str) -> Iterator[Tuple[str, str]]:
        """
        fetches mappings
        """
        from_vals = list(from_vals)
        if not from_vals:
            return
        for store in self.stores:
            ors = [{from_field: from_val} for from_val in from_vals]
            if len(ors) == 1:
                where = ors[0]
            else:
                where = {"$or": ors}
            for obj, _, __ in store.find(where=where):
                from_val = obj.get(from_field, None)
                to_val = obj.get(to_field, None)
                if to_val:
                    yield from_val, to_val
                    from_vals.pop(from_val)


