# Contributing

Contributions (pull requests) are very welcome.

First fork the library on GitHub.

Then clone and install the library in development mode:

```bash
git clone https://github.com/your-username-here/torchtyping.git
cd torchtyping
pip install -e .
```

Then install the pre-commit hook:

```bash
pip install pre-commit
pre-commit install
```

These automatically check that the code is formatted, using Black and flake8.

Make your changes. Make sure to include additional tests testing any new functionality.

Verify the tests all pass:

```bash
pip install pytest
pytest
```

Push your changes back to your fork of the repository:

```bash
git push
```

Then open a pull request on GitHub.
