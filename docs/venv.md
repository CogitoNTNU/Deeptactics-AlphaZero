# Virtual environment in VSCode

## Setup

1. Create a virtual environment

    ```bash
    python -m venv venv
    ```

2. Activate the virtual environment

    ```bash
    source venv/bin/activate
    ```

3. Install the dependencies

    ```bash
    pip install -r requirements.txt
    ```

4. Deactivate the virtual environment

    ```bash
    deactivate
    ```

## Add new dependencies

1. Activate the virtual environment

    ```bash
    source venv/bin/activate
    ```

2. Install the new dependencies

    ```bash
    pip install <package>
    ```

3. Freeze the dependencies

    ```bash
    pip freeze > requirements.txt
    ```

4. Deactivate the virtual environment

    ```bash
    deactivate
    ```
