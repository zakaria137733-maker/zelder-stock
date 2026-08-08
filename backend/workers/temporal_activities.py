import os
import subprocess
import sys

from temporalio import activity


@activity.defn
async def free_collect_activity() -> None:
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("Running free collector (Google News + Yahoo Finance)...")
    subprocess.run([sys.executable, os.path.join("scripts", "free_collect.py")], cwd=backend_dir, check=True)
    print("Free collection complete.")
