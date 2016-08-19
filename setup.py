"""A setuptools based setup module.
https://packaging.python.org/en/latest/distributing.html
"""

import setuptools
import os.path

# Get the long description from the README file
readme_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.rst')
with open(readme_file, mode='r', encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    # Name
    name='simulation',
    
    # Desctiption
    description='simulation functions',
    long_description=long_description,
    
    # Keywords
    keywords='simulation functions',

    # Homepage
    url='https://github.com/jor-/measurements',

    # Author
    author='Joscha Reimer',
    author_email='jor@informatik.uni-kiel.de',

    # Version
    use_scm_version=True,

    # License
    # license='MIT',

    # Classifiers
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
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
        #'License :: OSI Approved :: MIT License',

        # Supported Python versions
        'Programming Language :: Python :: 3',
        #'Programming Language :: Python :: 3.5',scikits
    ],


    # Packages to install
    packages=setuptools.find_packages(),

    # Dependencies
    setup_requires=[
        'setuptools>=0.8',
        'pip>=1.4',
        'setuptools_scm',
    ],
    install_requires=[
        'numpy',
        'utillib[cache,options,interpolate,cholmod]',
        'measurements',
    ],
    extras_require={
        'asymptotic' : ['scipy'],
        'sorted_measurements_dict': ['measurements[sorted_measurements_dict]'],
    },
    dependency_links = [
        'git+https://github.com/jor-/util.git#egg=utillib-0.1.dev90+n72d8d0a',
    ]
)
