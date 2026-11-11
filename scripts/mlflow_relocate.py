"""Make an MLflow store created in Colab usable on your laptop.

MLflow remembers the ABSOLUTE folder of the artifacts (for example /content/agroscan/mlartifacts). On your laptop that folder
does not exist, so we rewrite it.

Usage (from the folder that contains mlflow.db and mlartifacts/):
    python scripts/mlflow_relocate.py /content/agroscan/mlartifacts  "$(pwd)/mlartifacts"
"""
import sqlite3
import sys

old, new = sys.argv[1], sys.argv[2]
con = sqlite3.connect("mlflow.db")
for table, column in (("experiments", "artifact_location"), ("runs", "artifact_uri"), ("model_versions", "source")):
    try:
        con.execute(f"UPDATE {table} SET {column} = REPLACE({column}, ?, ?)", (old, new))
    except sqlite3.OperationalError as e:
        print("skipped", table, e)
con.commit()
print("done: artifact locations rewritten from", old, "to", new)
