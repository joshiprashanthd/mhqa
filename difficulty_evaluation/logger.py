import csv
from pathlib import Path
import datetime
from typing import List

def timestring():
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S")

class CsvLogger:
    def __init__(self, dir_path: Path, fileprefix: str, headers: List[str]):
        self.fileprefix = fileprefix
        self.file = open(dir_path / f"{fileprefix}_{timestring()}.csv", "w")
        self.writer = csv.writer(self.file)
        self.writer.writerow(headers)

    def logrow(self, values: List[str]):
        self.writer.writerow(values)
    
    def logrows(self, values: List[List[str]]):
        self.writer.writerows(values)

    def close(self):
        self.file.close()