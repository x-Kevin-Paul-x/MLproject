from setuptools import find_packages,setup
from typing import List
def get_requirements(file_name:str)->List[str]:
    requirements=[]
    with open(file_name) as f:
        requirements=f.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements
    

setup(
    name="mlproject",
    version="0.0.1",
    description="An Ml project.",
    author="Kevin Paul",
    author_email="kevinpaul7h@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)