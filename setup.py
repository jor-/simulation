# simulation: a collection of functions to handel simulations using Metos3D
# Copyright (C) 2011-2017  Joscha Reimer jor@informatik.uni-kiel.de
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""A setuptools based setup module.
https://packaging.python.org/en/latest/distributing.html
"""

import setuptools
import os.path
import versioneer

# Get the long description from the README file
readme_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.rst')
with open(readme_file, mode='r', encoding='utf-8') as f:
    long_description = f.read()

# Setup
setuptools.setup(
    # general informations
    name = 'simulation',
    description = 'simulation functions',
    long_description = long_description,
    keywords = 'simulation functions',

    url = 'https://github.com/jor-/measurements',
    author = 'Joscha Reimer',
    author_email = 'jor@informatik.uni-kiel.de',
    license = 'AGPLv3+',

    classifiers = [
        # Development Status
        'Development Status :: 3 - Alpha',
        # Intended Audience, Topic
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Libraries :: Python Modules',
        # Licence (should match "license" above)
        'License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)',
        # Supported Python versions
        'Programming Language :: Python :: 3',
    ],

    # version
    version=versioneer.get_version().replace('.dev0+', '+dirty.').replace('.post', '.post.dev'),
    cmdclass=versioneer.get_cmdclass(),

    # packages to install
    packages = setuptools.find_packages(),

    # dependencies
    setup_requires = [
        'setuptools>=0.8',
        'pip>=1.4',
    ],
    install_requires = [
        'numpy',
        'scipy',
        'matrix-decomposition',
        'utillib[cache,options,interpolate]',
        'measurements',
    ],
    extras_require = {
        'sorted_measurements_dict': ['measurements[sorted_measurements_dict]'],
    },

    # scripts
    entry_points={
        'console_scripts': [
            'simulation_model_check_integrity = simulation.model.check_integrity:_main',
            'simulation_model_save = simulation.model.save:_main',
            'simulation_model_remove = simulation.model.remove:_main',
            'simulation_optimization_save = simulation.optimization.save:_main',
            'simulation_optimization_matlab_cost_function_eval = simulation.optimization.matlab.cost_function:_main',
        ],
    }
)
