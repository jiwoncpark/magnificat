from setuptools import setup, find_packages

setup(
      name='magnificat',
      version='v0.10',
      author='Ji Won Park',
      author_email='jiwoncpark@gmail.com',
      packages=find_packages(),
      license='LICENSE.md',
      description='Simulating AGN light curves for LSST',
      long_description=open("README.rst").read(),
      long_description_content_type='text/markdown',
      url='https://github.com/jiwoncpark/magnificat',
      include_package_data=True,
      test_suite='nose.collector',
      tests_require=['nose'],
      classifiers=['Development Status :: 4 - Beta',
      'License :: OSI Approved :: BSD License',
      'Intended Audience :: Developers',
      'Intended Audience :: Science/Research',
      'Operating System :: OS Independent',
      'Programming Language :: Python'],
      keywords='physics'
      )
