import subprocess
import sys
from setuptools import setup  # Changed from distutils.core to setuptools
import os
import urllib.request

def download_weight_file(url, dest_path):
    if not os.path.exists(dest_path):
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        urllib.request.urlretrieve(url, dest_path)
        print(f"Downloaded {url} to {dest_path}")
    else:
        print(f"File {dest_path} already exists.")

setup(name='DinoCut',
      version='1.0',
      description='Image Processing Pipeline for Synthetic Data Generation',
      author='Sean Gleason',
      author_email='sgleason@nalaquq.com',
      url='https://github.com/Nalaquq/DINOcut',
      packages=['DINOcut'],  # Replace 'my_package' with the actual package name
     )

def run_command(command: str) -> None:
    try:
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        sys.exit(1)

def check_package_installed(package: str) -> bool:
    installed_packages = subprocess.check_output([sys.executable, "-m", "pip", "list"]).decode("utf-8")
    return package.lower() in installed_packages.lower()

def install_github_repo(url: str, commit_hash: str) -> None:
    repo_name = url.split('/')[-1].replace('.git', '')
    run_command(f"git clone {url}")
    run_command(f"cd {repo_name} && git checkout -q {commit_hash}")
    install_requirements(repo_name)
    run_command(f"cd {repo_name} && {sys.executable} -m pip install -q -e .")

def install_requirements(repo_name: str) -> None:
    requirements_path = repo_name
    run_command(f"{sys.executable} -m pip install -r {requirements_path}")

repository_url = "https://github.com/IDEA-Research/GroundingDINO.git"
commit_hash = "57535c5a79791cb76e36fdb64975271354f10251"
package_name = "GroundingDINO"

def main():
    weight_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    dest_path = os.path.join('sam', 'weights', 'sam_vit_h_4b8939.pth')
    download_weight_file(weight_url, dest_path)

if __name__ == "__main__":
    install_requirements("requirements.txt")
    if not check_package_installed(package_name):
        install_github_repo(repository_url, commit_hash)
    else:
        print(f"{package_name} is already installed.")
    main()



