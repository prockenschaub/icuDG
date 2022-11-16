from pathlib import Path
import pyarrow.parquet as pq

from . import Constants
from ..multicenter.data import ICUEnvironment

PAD_VALUE = 2
STA_VARS = ["Id", "Age", "Gender", "Unit1", "Unit2", "HospAdmTime"]
DYN_VARS = ["Id", "ICULOS", "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
       "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
       "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
       "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
       "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
       "Fibrinogen", "Platelets"]
OUTC_VARS = ["Id", "ICULOS", "SepsisLabel"]


class PhysioNetEnvironment(ICUEnvironment):
    """Data for a single training set (setA or set B) from the PhysioNet 2019 challenge
    https://physionet.org/content/challenge-2019/1.0.0/
    """
    def __init__(self, db, pad_to):
        self.db = db
        self.pad_to = pad_to

    def load(self, debug=False) -> None:
        """Load the PhysioNet 2019 challenge data and bring it into a format compatible with the `multicenter` task
        """
        df = pq.read_table(Path(Constants.data_dir, self.db, f'{self.db}.parquet'))
        df = df.to_pandas()

        data = {}
        data['sta'] = df[STA_VARS].drop_duplicates()
        data['sta'].set_index(["Id"], inplace=True)
        data['dyn'] = df[DYN_VARS]
        data['dyn'].set_index(["Id", "ICULOS"], inplace=True)
        data['outc'] = df[OUTC_VARS]
        data['outc'].set_index(["Id", "ICULOS"], inplace=True)
        
        if debug:
            # Limit to 1000 patients
            debug_stays = data["sta"].index.values[:1000]
            data = {k: v.loc[debug_stays, :] for k, v in data.items()}

        self.data = data

    def encode_categorical(self) -> None:
        """There are no categorical variables to encode for the PhysioNet 2019 Challenge"""
        self.ascertain_loaded()
