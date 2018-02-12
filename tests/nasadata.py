import pathlib
import parse
import pytest
from datetime import datetime
import pandas as pd
import numpy as np


@pytest.fixture(params=['A_MRA', 'B_ASP', 'C_BLN', 'D_HBN', 'E_IJK'])
def fxt_dataset(request):
    filepath = (
        pathlib.Path(__file__).absolute().parents[1]
        / 'data'
        / 'energy'
        / f'{ request.param }.txt'
    )

    df = pd.read_csv(
        filepath, delim_whitespace=True, skiprows=8, parse_dates=[[0, 1, 2]],
        date_parser=lambda *columns: datetime(*map(int, columns)))

    df.columns = ['date', 'exposure']
    df.set_index(df['date'], inplace=True)
    df.drop('date', 1, inplace=True)

    with open(filepath, 'r') as f:
        for _ in range(3):
            line = f.readline()
        latitude = parse.search('Latitude {:f}', line)[0] / 360.0 * 2 * np.pi

    return {'data': df, 'latitude': latitude}
