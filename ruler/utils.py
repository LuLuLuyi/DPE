from typing import Union, List
import json

def read_manifest(manifest: str) -> List[dict]:
    """
    Read manifest file

    Args:
        manifest (str or Path): Path to manifest file
    Returns:
        data (list): List of JSON items
    """

    data = []
    try:
        f = open(manifest, 'r', encoding='utf-8')
    except:
        raise Exception(f"Manifest file could not be opened: {manifest}")

    errors = []
    for line in f.readlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            errors.append(line)
            continue
        data.append(item)
    f.close()

    return data


def write_manifest(output_path: str, target_manifest: List[dict], ensure_ascii: bool = True):
    """
    Write to manifest file

    Args:
        output_path (str or Path): Path to output manifest file
        target_manifest (list): List of manifest file entries
        ensure_ascii (bool): default is True, meaning the output is guaranteed to have all incoming non-ASCII characters escaped. If ensure_ascii is false, these characters will be output as-is.
    """
    with open(output_path, "w", encoding="utf-8") as outfile:
        for tgt in target_manifest:
            json.dump(tgt, outfile, ensure_ascii=ensure_ascii)
            outfile.write('\n')
