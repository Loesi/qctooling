import os
from pathlib import Path


package_dir = Path(__file__).resolve().parent.parent
libcint_dir = package_dir / "_libcint"
binaries = []

if libcint_dir.is_dir():
    for library in libcint_dir.glob("libcint.dll"):
        binaries.append((str(library), "qctooling/_libcint"))

linked_library_paths = os.environ.get("QCTOOLING_LINKED_LIBARY_PATHS", "")
for directory in filter(None, linked_library_paths.split(os.pathsep)):
    for library in Path(directory).glob("*.dll"):
        binary = (str(library), "qctooling/_libcint")
        if binary not in binaries:
            binaries.append(binary)
