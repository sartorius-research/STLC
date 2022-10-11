from setuptools import setup

setup(
    name='stlc',
    version='0.1.0',    
    description='The Simulation Testbed for Liquid Chromatography (STLC) package implements variations of the general rate model. The aim is to provide a basis for the development of an easy to use package for chromatography modeling in python.',
    url='https://github.com/shuds13/pyexample',
    author='David Andersson',
    author_email='david.andersson@sartorius.com',
    license='GPLv3',
    packages=['stlc'],
    install_requires=['scipy',
                      'numpy',
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python'
    ],
)
