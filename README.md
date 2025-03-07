# Template Repository

This codebase is used as a template for all R&D repositories. If you want to use this repository as a template, please follow the instructions described below.

### We recommend running this from a machine that has display capabilities, as it will try to open a browser window to authenticate with Github.

# Prerequisites
1. `python >= 3.7,<3.12`
2. `uv`, `git` and `curl` installed on your machine
3. `aws` CLI installed and credentials configured (`aws sso login`)
3. [`copier`](https://copier.readthedocs.io/en/stable/)

## Creating a new repo based off the `rnd-template`

1. First, make sure you have `copier` installed in your Python environment
```python
uv tool install copier
```
2. Then, use `copier` to generate the new repository

```python
copier copy --trust --vcs-ref HEAD git@github.com:Synthesia-Technologies/rnd-template.git {your_folder}
```

- `{your_folder}` is the name of the folder, as it will appear locally on your computer. It does not influence the repository on Github

## Understanding the questions asked during the templating process

### 1. `What is your project name?`
- This influences the naming of your repository, Python Package and Docker Image

**Example**:
- If `project name` is `template_test` (only alphanumeric characters and `_` are allowed), then:
    - The Github Repo will be called `rnd-template-test`
    - The PyPi package name will be `synthesia-rnd-template-test`
    - The Docker Image name will be `synthesia-rnd-template-test`
    - When importing the package in a python script, the name will be `template_test`

### 2. `What is the name of the project author?`
- Simply your name, as it will appear in `pyproject.toml`

### 3. `Which dependency management tool would you like to use?`
- Here you can choose between `uv` and `conda`. We recommend using `uv`, as that is the company standard.

### 4. `Would you like to create a repo in Synthesia-Technologies github?`
- Most of the times, the answer should be `yes`, given that you'll want to store your code on Github

### 5. `What is the python version to use for the project?`
- We currently support `>=3.8,<3.12`. If possible, use the latest supported version

## What is the templating process doing?

1. Creates a virtual env for that project
2. Creates a Github Repo and pre-configures it
    - Adds the `mlops` Github Group with admin permissions to the repo
    - Adds the `research-shared` Github Group with triage permissions to the repo
    - Runs `uv sync --all-extras` to compile the dependencies in `pyproject.toml` into `uv.lock` with fixed versions
    - Pushes the changes to the repo and opens a new PR
    - Adds branch protections for the `main` branch

If all went well, you should see a message containing the name of the repo and of a PR that was just opened for you, that runs the CI/CD workflow as a last validation step


<br>

## CI/CD Workflows

We use `Github Actions` to run our CI/CD Pipelines and we store the actual workflows in https://github.com/Synthesia-Technologies/gha-workflows and call them from there. This allows us to easily push updates that will be propagated across all repositories.

However, there's nothing preventing you from creating new workflows locally, as long as they're stored in `.github/workflows` (check this [guide](https://docs.github.com/en/actions/using-workflows/reusing-workflows)).

We propose the following approach:

1. On every push to an open PR (`.github/workflows/on_pr.yml`)
    - Run `pre-commit` and `pytest` on the latest code
    - Run `hadolint` on the latest `Dockerfile`s
    - Based on [semantic versioning](https://semver.org/), we derive what the next version should be. We use the [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) standard on top of this to format and detect versions.
        - e.g. if current version is 0.4.1 (based off the list of git `tags`) and:
            - any commit starts with `fix:` -> `0.4.2`
            - any commit starts with `feat:` -> `0.5.0`
            - any commit contains `BREAKING CHANGE:` -> `1.0.0`
            - by default, by not specifying any of these types of messages, it defaults to patch (-> `0.4.2`)
    - Edit `pyproject.toml` to set that version to `{version}.dev0+{commit_hash}`
    - Build and Push the Python code to CodeArtifact
        - with the version we just edited in `pyproject.toml`
    - Edit `pyproject.toml` again to set the version to `{version}` instead
        - We then push this change back to the branch
    - Build and Push the Docker Image to ECR (based on the previously mentioned Python Package)
        - with the tag `{version}.dev0-{commit_hash}`

2. On every push to `main` - should be only on PRs being merged -  (`.github/workflows/on_push_main.yml`)
    - Runs a `sanity check` that verifies whether the `pyproject.toml` version matches up with the `upcoming release tag`
    - Run `pre-commit` and `pytest` on the latest code
    - Build and push a new wheel of the package in `src/` to CodeArtifact, with the version set to `{version}`
    - Build and push a new Docker Image
        - By default, this fetches the Python Package with the version that was just pushed in the previous step and installs it
        - Pushes with tags `stable, latest`
            - You can alter this to your liking
    - Create a new release tag and new release.
        - This is based on the latest `tag`

Alternative Github Actions events could be:

```
# Trigger when something is pushed to dev
on:
    push:
        branches
            - 'dev'
```

```
# Trigger when something is pushed to anything but main
on:
    push:
        branches
            - '*'
            - '!main'
```

You can check all possible events [here](https://docs.github.com/en/actions/using-workflows/events-that-trigger-workflows#pull_request).

### Using your own base image for workflows

- In many cases, teams will want to use their own base image for their workflows to speed up the CI, because you can pre-install many dependencies in that image, so you don't have to do it for every run of the workflow.

For this reason, the main workflows are defined locally:
    - `build_and_push_docker_image.yml` - Builds and pushes Docker Images.
    - `python_lint_and_test.yml` - Runs `pre-commit` and `pytest`.
    - `push_python_package.yml` - Handles versioning and pushes the Python Package to CA.

These jobs use by default a base image with AWS Cli, Docker and Python pre-installed (`812634467557.dkr.ecr.eu-west-1.amazonaws.com/gha-rnd-runner-python-base:py{version}-latest`). But you can change that accordingly to your image of choice by altering the `container:` attribute.

### Controlling the hardware on which the workflows run

The hardware is controlled by labels, that are configured using the `runs-on:` attribute. They are all configured using the labels `[self-hosted, research]`, which tells GHA to run these on the `research` machine types.

If you want to change the machine type, you have 2 options:

- **research**:
    - Default machine, runs on one of instance types [`r5.xlarge`, `r5n.xlarge`, `r6i.xlarge`]
    - Storage size: 128GBs
    - There is 1 instance running non-stop
    - There is an additional instance spinning up during business hours (Mon-Fri 8:00-20:00)

- **research-gpu**:
    - GPU machine, runs on instance type `g4dn.2xlarge`
    - Storage size: 128GBs
    - There is 1 instance spinning up during business hours (Mon-Fri 8:00-20:00)

In case more machines are needed (depending on the load), the runners can scale up to 6 instances. The jobs are placed in a queue and executed sequentially.

### Pushing Docker Images manually

If you want to build Docker Images manually, you can use the `docker/build.py` script for it (params are documented within.)

There are 2 operating modes, controlled by the presence of the `--local` flag:
- When present, it uses the `image-local` build step, which means that the code is copied from local into the machine and pip installed
- When not present, it installs the package from `CodeArtifact`

If you want to modify what code is copied when the `--local` flag is set, check the `prepare_artifacts()` method in `build.py`

## Way of Working
To ensure a consistent coding style we follow PEP8 best practices, for which you can find more information [here](https://www.python.org/dev/peps/pep-0008/). In order to keep a well maintained repository we allow both Gitflow and Trunk best practices, for which you can find more information [here](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow).


## Commiting to GitHub
Please **run `pre-commit install`** after cloning this repository. We use pre-commits to ensure that our coding style stays consistent and that we don't commit any files we don't want to. Have a read [here](https://pre-commit.com/) for more information. Whenever you try to commit, pre-commit will run the git hooks setup in `.pre-commit-config.yaml` and will notify you if any fail. You can also test pre-commit before committing by running `pre-commit run --all-files`. To ignore pre-commit you can run `git commit --no-verify`, however this is not advised unless necessary.

### Pre-commit steps
All the steps are specified in `.pre-commit-config.yaml`:
- Run common hooks for newlines, config etc.
- License Checker
- [`ruff`](https://github.com/charliermarsh/ruff)
- [uv lock](https://docs.astral.sh/uv/concepts/projects/#project-lockfile) - for pinning dependencies.

## Running Tests
Place all your tests under the `./tests` directory. Ensure the files are prefixed with `test_`, for example `test_gradients`. Also ensure that all the test methods are prefixed with `test_`, for example `def test_train_step()`. To run the testing suite, do the following:

```bash
$ pytest
```

## Installing dependencies
During development, we recommend installing the package in editable mode and including dev dependencies.
```bash
$ uv sync --all-extras
```

### Copyright
Copyright (c) 2024 Synthesia Limited - All Rights Reserved.
Unauthorized copying of this file, via any medium is strictly prohibited. Proprietary and confidental.
