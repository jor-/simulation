# simulation: a collection of functions to handel simulations using Metos3D
# Copyright (C) 2011-2016  Joscha Reimer jor@informatik.uni-kiel.de
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

# Get the long description from the README file
readme_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.rst')
with open(readme_file, mode='r', encoding='utf-8') as f:
    long_description = f.read()

# Version string
def version():
    import setuptools_scm
    def empty_local_scheme(version):
        return ""
    return {'local_scheme': empty_local_scheme}


setuptools.setup(
    # Name
    name = 'simulation',
    
    # Desctiption
    description = 'simulation functions',
    long_description = long_description,
    
    # Keywords
    keywords = 'simulation functions',

    # Homepage
    url = 'https://github.com/jor-/measurements',

    # Author
    author = 'Joscha Reimer',
    author_email = 'jor@informatik.uni-kiel.de',

    # Version
    use_scm_version = version,

    # License
    license = 'GPLv3+',

    # Classifiers
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers = [
        # Development Status
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Intended Audience, Topic
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Libraries :: Python Modules',

        # Licence (should match "license" above)
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',

        # Supported Python versions
        'Programming Language :: Python :: 3',
    ],

    # Packages to install
    packages = setuptools.find_packages(),

    # Dependencies
    setup_requires = [
        'setuptools>=0.8',
        'pip>=1.4',
        'setuptools_scm',
    ],
    install_requires = [
        'numpy',
        'utillib[cache,options,interpolate,cholmod]',
        'measurements',
    ],
    extras_require = {
        'asymptotic' : ['scipy'],
        'sorted_measurements_dict': ['measurements[sorted_measurements_dict]'],
    },
)
