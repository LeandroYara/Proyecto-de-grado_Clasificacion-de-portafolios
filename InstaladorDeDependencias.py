import os
import subprocess
import sys
# implement pip as a subprocess:

import subprocess

def install_dependencies():
    try:
        # Use subprocess to run the pip install command
        subprocess.check_call(['pip', 'install', '-r', 'requirements.txt'])
        print("Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to install dependencies. {e}")

# Call the function to install dependencies
install_dependencies()