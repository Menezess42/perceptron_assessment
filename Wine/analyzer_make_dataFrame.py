import json
import os
import re
from pathlib import Path

import pandas as pd


class Analyzer_test_1:
    def __init__(self, base_dir="./Reports/test_1/"):
        self.base_dir = Path(base_dir)
        self.data = []

    def load_all(self):
        for weight_dir in sorted(self.base_dir.glob("weights_*")):
            weight_key = weight_dir.name.replace("weights_", "")
            for run_dir in sorted(weight_dir.glob("run_*")):
                run_index = int(run_dir.name.replace("run_", ""))
                record = {
                    "weight_init": weight_key,
                    "run": run_index,
                }

                best_model_file = next(run_dir.glob("best_model_epoch_*.json"), None)
                if best_model_file:
                    match = re.search(
                        r"best_model_epoch_(\d+)\.json", best_model_file.name
                    )
                    if match:
                        record["best_model_epoch"] = int(match.group(1))
                    with open(best_model_file, "r") as f:
                        record["best_model"] = json.load(f)

                model_file = run_dir / "model.json"
                if model_file.exists():
                    with open(model_file, "r") as f:
                        record["final_model"] = json.load(f)

                test_report_file = run_dir / "test_report.json"
                if test_report_file.exists():
                    with open(test_report_file, "r") as f:
                        test_data = json.load(f)
                        record.update(
                            {
                                "accuracy": test_data.get("accuracy"),
                                "avg_cross_entropy_loss": test_data.get(
                                    "avg_cross_entropy_loss"
                                ),
                                "total_samples": test_data.get("total_samples"),
                                "predictions": test_data.get("predictions"),
                            }
                        )

                validation_file = run_dir / "validation_log.json"
                if validation_file.exists():
                    with open(validation_file, "r") as f:
                        record["validation_log"] = json.load(f)

                self.data.append(record)

    def to_dataframe(self):
        return pd.DataFrame(self.data)

    def save_dataframe(self, path=None):
        df = self.to_dataframe()
        path = path or self.base_dir / "compiled_data_test_1.pkl"
        df.to_pickle(path)

    def load_dataframe(self, path=None):
        path = path or self.base_dir / "compiled_data.pkl"
        return pd.read_pickle(path)


class Analyzer_test_2:
    def __init__(self, base_dir="./Reports/test_2/"):
        self.base_dir = Path(base_dir)
        self.data = []

    def load_all(self):
        for lr_dir in sorted(self.base_dir.glob("lr_*")):
            lr_key = lr_dir.name.replace("lr_", "")
            for run_dir in sorted(lr_dir.glob("run_*")):
                run_index = int(run_dir.name.replace("run_", ""))
                record = {
                    "lr_init": lr_key,
                    "run": run_index,
                }

                best_model_file = next(run_dir.glob("best_model_epoch_*.json"), None)
                if best_model_file:
                    match = re.search(
                        r"best_model_epoch_(\d+)\.json", best_model_file.name
                    )
                    if match:
                        record["best_model_epoch"] = int(match.group(1))
                    with open(best_model_file, "r") as f:
                        record["best_model"] = json.load(f)

                model_file = run_dir / "model.json"
                if model_file.exists():
                    with open(model_file, "r") as f:
                        record["final_model"] = json.load(f)

                test_report_file = run_dir / "test_report.json"
                if test_report_file.exists():
                    with open(test_report_file, "r") as f:
                        test_data = json.load(f)
                        record.update(
                            {
                                "accuracy": test_data.get("accuracy"),
                                "avg_cross_entropy_loss": test_data.get(
                                    "avg_cross_entropy_loss"
                                ),
                                "total_samples": test_data.get("total_samples"),
                                "predictions": test_data.get("predictions"),
                            }
                        )

                validation_file = run_dir / "validation_log.json"
                if validation_file.exists():
                    with open(validation_file, "r") as f:
                        record["validation_log"] = json.load(f)

                self.data.append(record)

    def to_dataframe(self):
        return pd.DataFrame(self.data)

    def save_dataframe(self, path=None):
        df = self.to_dataframe()
        path = path or self.base_dir / "compiled_data_test_2.pkl"
        df.to_pickle(path)

    def load_dataframe(self, path=None):
        path = path or self.base_dir / "compiled_data_test_2.pkl"
        return pd.read_pickle(path)


if __name__ == "__main__":
    analyzer_t1 = Analyzer_test_1()
    analyzer_t1.load_all()
    analyzer_t1.save_dataframe(path='./Reports/reports_test_1/compiled_data_test_1.pkl')
    analyzer_t2 = Analyzer_test_2()
    analyzer_t2.load_all()
    analyzer_t2.save_dataframe(path='./Reports/reports_test_2/compiled_data_test_2.pkl')
