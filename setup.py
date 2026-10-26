"""Bundle the same offline assets in wheels that the source checkout uses."""
from pathlib import Path
import shutil
from setuptools import setup
from setuptools.command.build_py import build_py

class BuildWithAssets(build_py):
    def run(self):
        super().run()
        destination=Path(self.build_lib)/'realityforge'/'assets'
        for folder in ('web','models','samples'):
            shutil.copytree(folder,destination/folder,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        for filename in ('NOTICE.md','LICENSE'):
            shutil.copy2(filename,destination/filename)

setup(cmdclass={'build_py':BuildWithAssets})
