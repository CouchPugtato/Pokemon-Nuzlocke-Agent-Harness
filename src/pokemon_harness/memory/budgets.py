"""persisting conservative usage calibration on the controlling thread"""

import json
import math


class BudgetCalibration:
    def __init__(self, connection):
        self.connection = connection

    def factor(self, key):
        row = self.connection.execute("SELECT value FROM metadata WHERE key=?", ("context_budget:" + key,)).fetchone()
        return json.loads(row[0])["factor"] if row else 1.0

    def record(self, key, report, observation_id):
        factor = report.get("calibration_suggestion")
        if not isinstance(factor, (int, float)) or isinstance(factor, bool) or not math.isfinite(factor) or factor < 1:
            return
        factor = max(factor, self.factor(key))
        with self.connection:
            self.connection.execute("INSERT OR REPLACE INTO metadata VALUES(?,?)",
                                    ("context_budget:" + key, json.dumps({"factor": factor,
                                                                         "observation_id": observation_id})))
