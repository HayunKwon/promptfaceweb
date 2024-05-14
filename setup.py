import json
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().split("\n")

with open("package_info.json", "r", encoding="utf-8") as f:
    package_info = json.load(f)

setuptools.setup(
    name="promptface",
    version=package_info["version"],
    author="M1nu0x0",
    author_email="oxygen0112@naver.com",
    description=(
        "A custom version of deepface(0.0.91)"
    ),
    data_files=[("", ["README.md", "requirements.txt", "package_info.json"])],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/M1nu0x0/promptface",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "deepface = deepface.DeepFace:cli",
            "promptface = promptface.PromptFace:cli"],
    },
    python_requires=">=3.8",
    license="MIT",
    install_requires=requirements,
)