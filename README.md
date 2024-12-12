# How to Install and Update the Package #

## Create the package ##

### Step 1: Set Up the Conda Environment

1. Create the Conda Environment. To create the environment with the dependencies specified in environment.yml, navigate to the directory containing the file and run:
`conda env create -f environment.yml`

This will create a new environment called {env_name} with all the specified dependencies, including those installed via pip.

2. Activate the Environment. Once the environment is created, activate it with:
`conda activate {env_name}`


### Step 2: Install the package

After activating the {env_name} environment, you can install the package in “editable” mode, which allows you to make changes to the package code without reinstalling it.
Install the Package in Editable Mode. Run the following command from the directory containing the setup.py file:
`pip install -e .`

This installs the package specified by setup.py in editable mode, making it easy to make updates.


### Step 3: Verify the Installation
To check if the environment and package were installed correctly, you can import the package in Python:
`python -c "import codereil_ai_package_name"`

Replace codereil_ai_package_name with the actual name of the package as detected by find_packages().


## Update the Package ##

If you modify setup.py or the package itself, simply reinstall it in editable mode (as described above) by running:

`pip install -e .`



