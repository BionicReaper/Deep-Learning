import subprocess
import sys

# Run image2csv.py
subprocess.run(["python", "image2csv.py", sys.argv[1]], check=True)

# Compile
subprocess.run(["gcc", "-o", "deeplearning", "deeplearning.c"], check=True)

# Then run deeplearning.exe
subprocess.run(["deeplearning.exe", "--classify=output.csv"], check=True)