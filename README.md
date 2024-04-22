[![Unit Tests](https://github.com/mpm-tu-berlin/eflips-eval/actions/workflows/unittests.yml/badge.svg)](https://github.com/mpm-tu-berlin/eflips-eval/actions/workflows/unittests.yml) 
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# eflips-eval

---

Part of the [eFLIPS/simBA](https://github.com/stars/ludgerheide/lists/ebus2030) list of projects.

---


This repository contains code to evaluate an eflips/simBA simulation. It provides the `eflips.eval`package, which
contains functions to prepare (database -> pandas DataFrame) and visualize (pandas DataFrame -> plotly) simulation results.

It is organized hierarchically as follows:
- `eflips.eval.input` contains functions to prepare and visualize simulation input data. That means it does not access
  any data from the `Event` or `Vehicle` classes.
- `eflips.eval.output` contains functions to prepare and visualize simulation output data. That means these functions 
  all access the `Event` or `Vehicle` classes.
- within each package, there is a `prepare` and a `visualize` module. The `prepare` module contains functions to prepare
  the data, the `visualize` module contains functions to visualize the data. The functions must conform tp a naming
  convention, in which for the `eflips.eval.input.preapre.foo()`, there must also exist a
  `eflips.eval.input.visualize.foo()` method and vice versa. However, when the exact same data can be visualized in 
  multiple ways, there can also be a `eflips.eval.input.visualize.foo_scatter()` and a 
  `eflips.eval.input.visualize.foo_hist()` method.
- the signature of a `prepare` method is always `prepare(identifier(s), session) -> pd.DataFrame`, where `identifier(s)`
  is a single or tuple of `int`s identifying a specific database object and `session` is a 
  `sqlalchemy.orm.session.Session` object that is used to access the database.
- the signature of a `visualize` method is always `visualize(df: pd.DataFrame, **kwargs) -> plotly.graph_objects.Figure`,
  where `df` is the DataFrame that is to be visualized and `kwargs` are additional arguments that can be used to
  customize the visualization. it does however create a legend (if necessary) and sets axis labels.
- a visualize function does not make assumptions about the size of the viewport and does not set the title of the plot.
- The resulting plots should be in english, with proper names (e.g. "Distance [km]" instead of `total_dist`) for the
  values that would be shown to the user.

## Installation

**TODO**

## Usage

**TODO**

## Testing

**TODO**

### Documentation

The documentation is generated using [sphinx](https://www.sphinx-doc.org/en/master/). To generate the documentation,
execute the following command in the root directory of the repository:

```bash
sphinx-build doc/ doc/_build -W
```

### Development

We utilize the [GitHub Flow](https://docs.github.com/get-started/quickstart/github-flow) branching structure. This means  that the `main` branch is always deployable and that all development happens in feature branches. The feature branches are merged into `main` via pull requests. We utilize the [semantic versioning](https://semver.org/) scheme for versioning.

Dependencies are managed using [poetry](https://python-poetry.org/). To install the dependencies, execute the following command in the root directory of the repository:

```bash
poetry install
```

We use black for code formatting. You can use `black .` to format the code.

We use [MyPy](https://mypy.readthedocs.io/en/stable/) for static type checking. You can
use ` mypy --strict --explicit-package-bases  eflips/` to run MyPy on the code.

Please make sure that your `poetry.lock` and `pyproject.toml` files are consistent before committing. You can use `poetry check` to check this. This is also checked by pre-commit.

You can use [pre-commit](https://pre-commit.com/) to ensure that MyPy, Black, and Poetry are run before committing. To
install pre-commit, execute the following command in the root directory of the repository:

We recommend utilizing linters such as [PyLint](https://pylint.readthedocs.io/en/latest/index.html) for static code
analysis (but not doing everything it says blindly).

## Usage Example

In [examples](examples/) an example script can be found that generates a report.

## License

This project is licensed under the AGPLv3 license - see the [LICENSE](LICENSE.md) file for details.

## Funding Notice

This code was developed as part of the project [eBus2030+](https://www.eflip.de/) funded by the Federal German Ministry for Digital and Transport (BMDV) under grant number 03EMF0402.


