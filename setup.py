from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='pyautoplot',
    version='{{VERSION_PLACEHOLDER}}',
    author='Infinitode Pty Ltd',
    author_email='infinitode.ltd@gmail.com',
    description="PyAutoPlot is an open-source Python library designed to make dataset analysis much easier by generating helpful detailed plots using matplotlib based on your dataset.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/infinitode/pyautoplot',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
    ],
    python_requires='>=3.6',
    install_requires=[
        'matplotlib>=3.0.0',
        'pandas>=1.0.0',
        'numpy>=1.18.0'
    ],
)
