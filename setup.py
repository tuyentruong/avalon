import os
import sys

from setuptools import Command, setup
from setuptools.dist import Distribution


class InstallCommand(Command):
    """Performs a clean installation of package dependencies using Pipenv."""
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        os.system('conda env remove -n avalon -y')
        if sys.platform == 'darwin':
            os.system('brew install coreutils')
        os.system('conda env create -n avalon --file environment.yaml')
        os.system('conda run -n avalon python -m avalon.install_godot_binary --overwrite')
        os.system('conda run -n avalon python -m avalon.common.check_install')


class BinaryDistribution(Distribution):
    """Distribution which always forces a binary package with platform name"""

    def has_ext_modules(self) -> bool:
        return True


setup(
    # Include pre-compiled extension
    # package_data={"packagename": ["_precompiled_extension.pyd"]},
    # distclass=BinaryDistribution,
    cmdclass={
        'install': InstallCommand
    }
)
