1. Download the dataset 
Link: https://zenodo.org/record/7893847

2. Python
Download and install Python version 3.10 (https://www.python.org/downloads/release/python-31011/).

If you had a version of Python installed already, make sure you are using the correct version e.g.
```
~code>python --version
Python 3.10.11
```
(Any Python 3.10 and higher version will most likely work, however during development only 3.10.11 was used)

Create a new Python virtual environment.
```bash
python -m venv venv
```
Activate the virtual environment.

- On Windows 
```
venv\Scripts\activate
```
- On Linux
```
source venv/bin/activate
```
Install required Python packages from requirements.txt.
```bash
pip install -r requirements.txt
```

3. Run the project
After you are done installing the project dependencies simply run the project using
```bash
python src/main.py [-f|--config_file <path_to_config>]
```

4. Configuration
Configuration is done via a INI file (Or by modifying the source code itself if you so desire).
Refer to the default *config.ini* file to see the different configuration options. 
