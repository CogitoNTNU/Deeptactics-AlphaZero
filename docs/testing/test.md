# Testing the code

## How to run the tests

1. Navigate to the root of the project

2. If you haven't already, install pytest by running the following command in the terminal:

    ```bash
    pip install pytest
    ```

3. Run the tests by running the following command in the terminal:

    ```bash
    pytest
    ```

## What to expect

If all tests pass, you should see an output which says that all tests passed.
If any tests fail, you might want to run in debug mode to see what went wrong.

## Debugging

To run the tests in debug mode, run the following command in the terminal:

```bash
pytest --pdb
```

You can now print variables to see what is wrong.
