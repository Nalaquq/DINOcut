import subprocess
import sys


def run_command(command: str) -> None:
    """
    Runs a shell command and handles exceptions.

    Parameters:
        command (str): The command to be executed in the shell.

    Raises:
        SystemExit: Exits the script with status code 1 if the command fails.
    """
    try:
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        sys.exit(1)

def check_package_installed(package: str) -> bool:
    """
    Checks if a specified Python package is installed.

    Parameters:
        package (str): The name of the package to check.

    Returns:
        bool: True if the package is installed, False otherwise.
    """
    installed_packages = subprocess.check_output([sys.executable, "-m", "pip", "list"]).decode("utf-8")
    return package.lower() in installed_packages.lower()

def install_github_repo(url: str, commit_hash: str) -> None:
    """
    Clones a GitHub repository, checks out a specific commit, and installs the package.

    Parameters:
        url (str): The GitHub repository URL.
        commit_hash (str): The specific commit hash to check out.

    Description:
        This function first clones the repository from the provided URL, then checks out
        the specified commit. It installs the package using pip in editable mode and then
        attempts to install any dependencies listed in a `requirements.txt` file.
    """
    # Extract repository name from URL and remove ".git" if present
    repo_name = url.split('/')[-1].replace('.git', '')
    run_command(f"git clone {url}")
    run_command(f"cd {repo_name} && git checkout -q {commit_hash}")
    install_requirements(repo_name)
    run_command(f"cd {repo_name} && {sys.executable} -m pip install -q -e .")

def install_requirements(repo_name: str) -> None:
    """
    Installs the dependencies listed in the requirements.txt file of the cloned repository.

    Parameters:
        repo_name (str): The name of the repository directory where the requirements.txt is located.

    Raises:
        SystemExit: Exits the script with status code 1 if the installation fails.
    """
    requirements_path = repo_name
    run_command(f"{sys.executable} -m pip install -r {requirements_path}")

# Main execution logic
repository_url = "https://github.com/IDEA-Research/GroundingDINO.git"
commit_hash = "57535c5a79791cb76e36fdb64975271354f10251"
package_name = "GroundingDINO"  # Assuming this is the correct package name

#install_requirements("requirements.txt")

if not check_package_installed(package_name):
    install_github_repo(repository_url, commit_hash)
else:
    print(f"{package_name} is already installed.")
