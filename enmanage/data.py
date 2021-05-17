from pkg_resources import resource_filename
import pandas as pd
import numpy as np
import parse
from datetime import datetime


class ExampleData(object):
    @staticmethod
    def load_dataset(filepath):
        df = pd.read_csv(
            filepath,
            delim_whitespace=True,
            skiprows=8,
            parse_dates=[[0, 1, 2]],
            date_parser=lambda *columns: datetime(*map(int, columns)),
        )

        df.columns = ["date", "exposure"]
        df.set_index(df["date"], inplace=True)
        df.drop("date", 1, inplace=True)

        with open(filepath, "r") as f:
            for _ in range(3):
                line = f.readline()
            latitude = parse.search("Latitude {:f}", line)[0] / 360.0 * 2 * np.pi

        return {"data": df, "latitude": latitude}

    def __getitem__(self, key):
        if key not in ["A_MRA", "B_ASP", "C_BLN", "D_HBN", "D_SOL", "E_IJK"]:
            raise ValueError(f"Unknown dataset {key}")

        return ExampleData.load_dataset(resource_filename(__name__, f"data/{key}.txt"))


data = ExampleData()
