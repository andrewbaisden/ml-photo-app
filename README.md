# ML Photo App

## Install and Setup

This project uses [virtualenv](https://pypi.org/project/virtualenv/) for creating a virtual python environment.

### neural-network-builder

`cd` into the folder `neural-network-builder`.

Run the commands below, to setup the virtual environment:

```python
virtualenv venv
source venv/bin/activate
cd venv

# Use deactivate to deactivate the virtual environment
deactivate
```

Install the `requirements.txt` file for your version of Python.

```python
# Python 3 version
pip3 install -r requirements.txt
```

```python
# Python 2 version
pip install -r requirements.txt
```

#### Running the neural-network-builder

Run this command:

```shell
# Python 3
python3 generate-model.py

# Python 2
python generate-model.py
```

### Frontend

`cd` into the folder `frontend`.

Run the command below, to setup the virtual environment:

```python
virtualenv venv
source venv/bin/activate
cd venv

# Use deactivate to deactivate the virtual environment
deactivate
```

Install the `requirements.txt` file for your version of Python.

```python
# Python 3 version
pip3 install -r requirements.txt
```

```python
# Python 2 version
pip install -r requirements.txt
```

#### Running the Frontend

Run this command:

```shell
taipy run index.py
```
