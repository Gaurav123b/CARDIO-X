"""
Environment verification script for Cardio-X project.
Performs non-destructive checks on Python interpreter, paths, and dependencies.
"""

import sys
from pathlib import Path

def run_verification():
    print("=" * 60)
    print("CARDIO-X ENVIRONMENT VERIFICATION REPORT")
    print("=" * 60)
    
    # 1. Interpreter Check
    print(f"[✓] Python Executable : {sys.executable}")
    print(f"[✓] Python Version    : {sys.version.split()[0]}")
    
    # 2. Virtual Environment Check
    in_venv = sys.prefix != sys.base_prefix
    print(f"[✓] Inside Virtualenv : {'Yes' if in_venv else 'No'} ({sys.prefix})")
    
    # 3. Project Directory Structure
    root = Path(__file__).resolve().parent
    expected_dirs = ["data", "notebooks", "src", "models", "reports", "tests"]
    all_dirs_ok = True
    for d in expected_dirs:
        dir_path = root / d
        exists = dir_path.is_dir()
        if not exists:
            all_dirs_ok = False
        print(f"[{'✓' if exists else '✗'}] Directory '{d}/': {'Exists' if exists else 'Missing'}")
    
    # 4. Key Files Check
    expected_files = [".gitignore", "requirements.txt", "README.md", "data/README.md", "src/__init__.py"]
    all_files_ok = True
    for f in expected_files:
        file_path = root / f
        exists = file_path.is_file()
        if not exists:
            all_files_ok = False
        print(f"[{'✓' if exists else '✗'}] File '{f}': {'Exists' if exists else 'Missing'}")
        
    # 5. Dependency Import Check (optional check for installed packages)
    packages = ["pandas", "numpy", "matplotlib", "seaborn", "sklearn", "pytest"]
    print("\nPackage Import Status:")
    for pkg in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "installed")
            print(f"  [✓] {pkg:<12}: {version}")
        except ImportError:
            print(f"  [-] {pkg:<12}: Not yet installed (Pending approval)")
            
    print("=" * 60)
    if in_venv and all_dirs_ok and all_files_ok:
        print("SUCCESS: Cardio-X project structure and virtual environment verified!")
    else:
        print("WARNING: Some verification checks require attention.")
    print("=" * 60)

if __name__ == "__main__":
    run_verification()
