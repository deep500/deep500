import csv
import os

from ..event import RunnerEvent
from ..summaries import TrainingStatistics



class CsvWriterEvent(RunnerEvent):
    def __init__(self, csv_name: str = "training_stats", folder: str = os.getcwd()):
        super(CsvWriterEvent, self).__init__()
        self.test_csv = os.path.join(folder, csv_name + "_" + "test.csv")
        self.train_csv = os.path.join(folder, csv_name + "_" + "train.csv")

    def _to_csv(self, filename, df):
        with open(filename, df) as f:
            if df is None or len(df) == 0:
                return

            w = csv.DictWriter(f, df[0].keys())
            w.writeheader()
            for row in df:
                w.writerow(row)
                
        print('Written CSV job summary to {}'.format(filename))
        
    def after_training(self, runner, training_stats: TrainingStatistics):
        # current epoch should be the last used
        df_train, df_test = training_stats.to_dict()
        self._to_csv(self.train_csv, df_train)
        self._to_csv(self.test_csv, df_test)
