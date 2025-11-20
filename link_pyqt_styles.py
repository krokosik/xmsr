from PyQt5.QtCore import QLibraryInfo
import os
from pathlib import Path

print("Qt plugin paths:")
print(QLibraryInfo.location(QLibraryInfo.PluginsPath))
print("QT_PLUGIN_PATH:", os.environ.get("QT_PLUGIN_PATH"))

plugin_path = Path(QLibraryInfo.location(QLibraryInfo.PluginsPath)) / "styles"
src_path = Path(os.environ.get("QT_PLUGIN_PATH", "")) / "styles"

if not plugin_path.exists() and src_path.exists():
    print(f"Linking {src_path} to {plugin_path}")
    os.symlink(src_path, plugin_path)