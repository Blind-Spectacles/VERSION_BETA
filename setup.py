from setuptools import setup, find_packages

setup(
    name="object_detection_package",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow",
        "opencv-python",
        "numpy"
    ],
    description="Object detection and distance estimation using MobileNet SSD",
    author="Your Name",
    author_email="your_email@example.com",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
