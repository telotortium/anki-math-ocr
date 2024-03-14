from setuptools import setup, find_packages

setup(
    name="anki-math-ocr",
    version="0.1.0",
    author="Robert Irelan",
    author_email="rirelan@gmail.com",
    description="OCR Anki images using Google Gemini Pro",
    packages=find_packages(),
    install_requires=[
        "google-generativeai",
        "markdown",
        "pillow",
        "requests",
    ],
    entry_points={
        "console_scripts": [
            "anki-math-ocr=anki_math_ocr.__init__:main",
        ],
    },
)
