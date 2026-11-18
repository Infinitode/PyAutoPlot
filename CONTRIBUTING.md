# Contributing to PyAutoPlot

Thank you for considering contributing to **PyAutoPlot**! Contributions, whether they are bug fixes, new features, or improvements to the documentation, are always welcome. Before contributing, please read the following guidelines to ensure smooth collaboration.

## Getting Started

1. **Fork the Repository**
   Create a fork of this repository to your GitHub account and clone it locally.

2. **Set Up the Environment**
   Ensure you have Python 3.6+ installed. You should also install compatible versions of the necessary dependencies for the project.

3. **Run Tests**
   Install the package and run the headless regression suite:

   ```bash
   python -m pip install .
   python -m unittest discover -s tests -v
   ```

   `python test.py` is a separate smoke example using the included energy dataset.

## How to Contribute

1. **Report Issues**
   Use the [GitHub Issues page](https://github.com/Infinitode/PyAutoPlot/issues) to report bugs or suggest features. Please include:
   - A clear description of the issue or suggestion.
   - Steps to reproduce (for bugs).

2. **Make Changes**
   - Use clear and descriptive commit messages.
   - (Optional) Write tests to show new functionality.
   - Ensure the code is **readable**. You can learn more about Python code readability here: https://peps.python.org/pep-0008/.

3. **Submit a Pull Request**
   - Push your changes to a feature branch in your fork.
   - Submit a pull request with a detailed explanation of what you've changed or added.
   - Ensure your PR passes all automated tests and adheres to the contribution guidelines.

## Publishing a release

In the [`pyautoplot` project's publishing settings](https://pypi.org/manage/project/pyautoplot/settings/publishing/), add a **GitHub** Trusted Publisher with these values:

| PyPI field | Value |
| --- | --- |
| Owner | `Infinitode` |
| Repository name | `PyAutoPlot` |
| Workflow name | `build_and_publish.yml` |
| Environment name | Leave blank |

The workflow does not use a GitHub environment or a PyPI API token. Because there is no environment approval step, limit GitHub write access to trusted maintainers and restrict who can create or modify release tags.

1. Update the version in `setup.py` and the release notes in `README.md`. Merge the release changes into `main` after CI passes.
2. On the merged release commit, create and push a matching numeric tag (no `v` prefix). For 1.1.1:

   ```bash
   git tag 1.1.1
   git push origin refs/tags/1.1.1
   ```

3. The **Publish to PyPI** workflow checks the tag against the built version and runs the regression suite against the wheel before publishing the wheel and source distribution. After publishing, follow the link in the workflow summary or the README's **PyPI provenance** badge to the **Security** tab on PyPI. Check that both files have a PyPI Publish attestation. Prior releases without attestations will not gain them retroactively.

## Community Guidelines

To maintain a positive and collaborative community, we ask that all contributors adhere to the following principles:

1. **Be Respectful**
   Treat others with respect, regardless of their background or expertise.

2. **Provide Constructive Feedback**
   Offer helpful and actionable suggestions during code reviews.

3. **Follow Licensing Requirements**
   Ensure that any derivative works comply with the [license](https://github.com/infinitode/pyautoplot/blob/main/LICENSE).

By contributing, you agree to abide by these guidelines and this project's [license](https://github.com/infinitode/pyautoplot/blob/main/LICENSE). Thank you for helping make **PyAutoPlot** better!
