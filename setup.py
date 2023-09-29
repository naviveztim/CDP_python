"""A python implementation of Concatenated-Decision-Path method for time series classification."""

from setuptools import find_packages, setup


NAME = 'cdp-ts'
DESCRIPTION = 'A python implementation of Concatenated-Decision-Path method for time series classification'
with open('README.md') as f:
    LONG_DESCRIPTION = f.read()
LONG_DESCRIPTION_CONTENT_TYPE = 'text/markdown'
MAINTAINER = 'Ivan Mitzev'
MAINTAINER_EMAIL = 'cdp_project@outlook.com'
URL = 'https://github.com/naviveztim/CDP_python'
LICENSE = 'new BSD'
DOWNLOAD_URL = 'https://github.com/naviveztim/CDP_python'
VERSION = 0.13
INSTALL_REQUIRES = ['numpy>=1.19.5',
                    'pandas>=1.1.5'
                    ]
CLASSIFIERS = ['Development Status :: 3 - Alpha',
               'Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 3.6',
               'Programming Language :: Python :: 3.8',
               'Programming Language :: Python :: 3.9',
               'Programming Language :: Python :: 3.10',
               'Programming Language :: Python :: 3.11']

setup(name=NAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      long_description=LONG_DESCRIPTION,
      long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
      zip_safe=False,
      classifiers=CLASSIFIERS,
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      )