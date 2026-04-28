from __future__ import annotations

import csv
import json
import os
from typing import Any, Iterable, List, Optional, Sequence, Union


class DataLoader:
    """Carga secuencias de elementos textuales desde archivos JSON o CSV."""

    def __init__(self, base_path: Optional[str] = None) -> None:
        self.base_path = os.path.abspath(base_path or os.getcwd())

    def _resolve_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.join(self.base_path, path)

    def load_csv(self, path: str, field_name: Optional[str] = None) -> List[str]:
        """Carga una lista de elementos desde un archivo CSV.

        Si se proporciona field_name, se extrae ese campo de cada fila. Si no,
        cada fila se convierte en una cadena completa.
        """
        full_path = self._resolve_path(path)
        elements: List[str] = []
        with open(full_path, encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file) if field_name else csv.reader(file)
            for row in reader:
                if field_name:
                    value = row.get(field_name)
                    if value is not None:
                        elements.append(str(value).strip())
                else:
                    row_text = " ".join(str(item).strip() for item in row if item is not None)
                    if row_text:
                        elements.append(row_text)
        return elements

    def load_json(self, path: str, key: Optional[str] = None) -> List[str]:
        """Carga una lista de elementos desde un archivo JSON.

        El JSON puede ser una lista de cadenas o una lista de objetos. Si se
        proporciona key, se extrae ese campo de cada objeto.
        """
        full_path = self._resolve_path(path)
        with open(full_path, encoding="utf-8") as file:
            payload = json.load(file)

        if isinstance(payload, list):
            if key is None:
                return [str(item).strip() for item in payload if item is not None]
            return [str(item.get(key, "")).strip() for item in payload if isinstance(item, dict)]

        if isinstance(payload, dict) and key is not None:
            value = payload.get(key)
            if isinstance(value, list):
                return [str(item).strip() for item in value if item is not None]

        raise ValueError("JSON content must be a list or contain a top-level list under the selected key.")

    def load(self, path: str, format: Optional[str] = None, **kwargs: Any) -> List[str]:
        """Carga datos de un archivo CSV o JSON según el formato especificado."""
        if format is None:
            extension = os.path.splitext(path)[1].lower()
            if extension == ".csv":
                format = "csv"
            elif extension == ".json":
                format = "json"
            else:
                raise ValueError("Formato no soportado. Use 'csv' o 'json'.")

        if format == "csv":
            return self.load_csv(path, field_name=kwargs.get("field_name"))
        if format == "json":
            return self.load_json(path, key=kwargs.get("key"))
        raise ValueError("Formato no soportado. Use 'csv' o 'json'.")


def flatten_sequence(items: Iterable[Union[str, Sequence[str]]]) -> List[str]:
    """Convierte una secuencia de elementos en una lista plana de cadenas."""
    flattened: List[str] = []
    for item in items:
        if isinstance(item, str):
            flattened.append(item.strip())
        else:
            flattened.extend(str(sub_item).strip() for sub_item in item if sub_item is not None)
    return flattened
