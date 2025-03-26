#!/usr/bin/env python3

# Copyright (c) 2025 Synthesia Limited - All Rights Reserved
#
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.

"""Build a docker image."""

import logging
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable, List, Optional

import boto3
import click
import yaml  # type: ignore
from botocore.exceptions import ClientError

SCRIPT_PATH = os.path.abspath(__file__)
SCRIPT_DIR = Path(SCRIPT_PATH).parent
ROOT_PATH = Path(SCRIPT_PATH).parent.parent
ECR_CONFIG_YAML_PATH = ROOT_PATH / "docker" / "ecr_config.yml"

logging.basicConfig(level=logging.INFO)


class DockerTargets(str, Enum):
    # Class must inherit from str in order to be compatible with click
    IMAGE_COMMON = "image-common"
    IMAGE_LOCAL = "image-local"
    IMAGE_PROD = "image-prod"


@dataclass
class S3CacheConfig:
    # Holds the S3 cache config and generates the corresponding Docker statement
    bucket: str
    docker_image_name: str
    ecr_url: str
    region: str

    def __str__(self):
        return f"type=s3,region={self.region},bucket={self.bucket},name={self.ecr_url}/{self.docker_image_name}"


def generate_temp_sso_credentials(file_path: Path):
    # Create a session using SSO profile
    session = boto3.Session()

    # Get temporary credentials from STS client
    sts_client = session.client("sts")
    response = sts_client.get_caller_identity()

    # Just as a sanity check, print the account and ARN details
    logging.info(f"Generating credentials from SSO Account: {response['Account']}, ARN: {response['Arn']}")

    # Extract credentials
    credentials = session.get_credentials()
    access_key = credentials.access_key
    secret_key = credentials.secret_key
    session_token = credentials.token

    with file_path.open("w") as f:
        f.write("[default]\n")
        f.write(f"aws_access_key_id = {access_key}\n")
        f.write(f"aws_secret_access_key = {secret_key}\n")
        f.write(f"aws_session_token = {session_token}\n\n")


def create_ecr_repository() -> None:
    """Creates an ECR repository if the config YAML is present."""
    if not ECR_CONFIG_YAML_PATH.exists():
        logging.info(f"ECR config file {ECR_CONFIG_YAML_PATH} does not exist. Skipping ECR repository creation.")
        return

    with ECR_CONFIG_YAML_PATH.open("r") as f:
        ecr_configs = yaml.safe_load(f)

    ecr_client = boto3.client("ecr")
    for ecr_config in ecr_configs:
        name = ecr_config["name"]
        tags = ecr_config["tags"]

        # Check if the repository already exists, and raise an exception if it does.
        try:
            response = ecr_client.describe_repositories(repositoryNames=[name])
            repositories = response["repositories"]
        except ClientError as exc:
            if exc.response["Error"]["Code"] != "RepositoryNotFoundException":
                raise exc

            logging.info(f"Creating ECR repository {name}")

            result = ecr_client.create_repository(
                repositoryName=name,
                tags=[{"Key": k, "Value": v} for k, v in tags.items()],
            )
            logging.info(f"ECR repository {name} created")
            return result
        else:
            logging.info(f"ECR repository {name} already exists, updating tags")
            ecr_client.tag_resource(
                resourceArn=repositories[0]["repositoryArn"],
                tags=[{"Key": k, "Value": v} for k, v in tags.items()],
            )


def command_to_string(cmd: List[str]) -> str:
    cmd = list(map(str, cmd))
    r = cmd.pop(0)
    for c in cmd:
        if c.startswith("-"):
            r += " \\\n" + c
        else:
            r += " " + c
    return r


def sanitize_command(command: Iterable[Any]) -> List[str]:
    if not isinstance(command, (list, tuple)):
        raise TypeError("We expect the command to be a list of arguments. Do not use single-string commands!")
    return list(map(str, command))


def run_command(cmd: List[str], verbose=False, **kwargs) -> str:
    """Run command with subprocess."""
    if verbose:
        print("[RUNNING] " + command_to_string(cmd))
    try:
        cmd_output = subprocess.check_output(sanitize_command(cmd), **kwargs)
        if isinstance(cmd_output, bytes):
            cmd_output = cmd_output.decode()
        return cmd_output
    except subprocess.CalledProcessError as e:
        if verbose:
            print("[ERROR] Subprocess error")
            if e.stderr is not None:
                print(e.stderr.decode("utf-8"))
        raise


def prepare_artifacts(path: Path, docker_target: Optional[DockerTargets]) -> None:
    """Prepares the artifacts to be copied to the docker context.

    Args:
        path (Path): Path to where the files should be stored.
        docker_target Optional(DockerTargets): Which docker image to build. This affectes whether the repo code
            and `requirements-dev.txt` are copied to the docker context or not.
    """
    if docker_target == DockerTargets.IMAGE_LOCAL:
        dest_path = path / "repo.tar.gz"
        with tarfile.open(dest_path, "w:gz") as tar:
            tar.add(ROOT_PATH / "src", arcname="src")
            tar.add(ROOT_PATH / "tests", arcname="tests")
            tar.add(ROOT_PATH / "pyproject.toml", arcname="pyproject.toml")
            if ROOT_PATH.joinpath("poetry.lock").exists():
                tar.add(ROOT_PATH / "poetry.lock", arcname="poetry.lock")
            if ROOT_PATH.joinpath("README.md").exists():
                tar.add(ROOT_PATH / "README.md", arcname="README.md")

        if ROOT_PATH.joinpath("requirements-dev.txt").exists():
            shutil.copy(ROOT_PATH / "requirements-dev.txt", path / "requirements-dev.txt")

    shutil.copy(ROOT_PATH / "pyproject.toml", path / "pyproject.toml")
    if ROOT_PATH.joinpath("uv.lock").exists():
        shutil.copy(ROOT_PATH / "uv.lock", path / "uv.lock")

    # Copy the requirements.txt file
    if ROOT_PATH.joinpath("requirements.txt").exists():
        shutil.copy(ROOT_PATH / "requirements.txt", path / "requirements.txt")


def parse_build_args_from_str(build_args: str) -> List[str]:
    """Parses the string of build args in order to obtain a list of args.

    Build arg names must contains only UPPERCASE letters, numbers and underscores. Value can be anything

    Args:
        build_args (str): Build args in string format

    Returns:
        List[str]: List of build args
    """
    parsed_build_args = []
    for extra_arg in build_args.split(","):
        arg_name, arg_value = extra_arg.strip().split("=")
        if not re.fullmatch("[A-Z0-9_]+", arg_name):
            raise ValueError(f"Argument name '{arg_name}' must be of form [A-Z0-9_]")

        parsed_build_args.append(f"{arg_name}={arg_value}")

    return parsed_build_args


def parse_docker_tags_from_str(docker_tags: str) -> List[str]:
    """Parses the string of docker tags in order to obtain a list of tags.

    Tags must contains only lowercase letters, numbers, dots underscores and dashes.

    Args:
        docker_tags (str): Docker tags in string format

    Returns:
        List[str]: List of docker tags args
    """
    parsed_docker_tags = []
    for docker_tag in docker_tags.split(","):
        docker_tag = docker_tag.strip()
        if not re.fullmatch("[a-z0-9._-]+", docker_tag):
            raise ValueError(f"Docker tag '{docker_tag}' must be of form [a-z0-9._]")

        parsed_docker_tags.append(docker_tag)

    return parsed_docker_tags


class DockerRegistryAccess:
    def __init__(self, ecr: str) -> None:
        self.ecr = ecr

    def __enter__(self):
        self.docker_registry_login(self.ecr)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.docker_registry_logout()

    @staticmethod
    def docker_registry_login(ecr: str):
        ps = subprocess.Popen(
            ["aws", "ecr", "get-login-password", "--region", "eu-west-1"],
            stdout=subprocess.PIPE,
        )
        run_command(
            ["docker", "login", "--username", "AWS", "--password-stdin", ecr],
            stdin=ps.stdout,
        )

    @staticmethod
    def docker_registry_logout():
        run_command(["docker", "logout"])


def image_exists_local(docker_image_name: str, docker_image_tag: str) -> bool:
    name_and_tag = None

    try:
        name_and_tag = f"{docker_image_name}:{docker_image_tag}"
        cmd_ret = run_command(["docker", "image", "inspect", name_and_tag], stderr=subprocess.PIPE)

        yaml_output = yaml.safe_load(cmd_ret)
        for yaml_entry in yaml_output:
            if name_and_tag in yaml_entry["RepoTags"]:
                return True
        return False
    except subprocess.CalledProcessError as e:
        error_message = e.stderr.decode("utf-8")

        # Different versions of docker return different error messages, canonize them
        error_message = error_message.replace("Error response from daemon:", "Error:")
        error_message = error_message.replace("no such image:", "No such image:")
        if f"Error: No such image: {name_and_tag}" in error_message:
            return False
        else:
            print("[ERROR] Subprocess error")

            print("-------------------------")
            print("Original error message:")
            print(e.stderr.decode("utf-8"))

            print("-------------------------")
            print("Processed error message:")
            print(error_message)
            raise


def image_exists_registry(docker_image_name: str, docker_image_tag: str, ecr: str) -> bool:
    with DockerRegistryAccess(ecr):
        remote_name_and_tag = f"{ecr}/{docker_image_name}:{docker_image_tag}"
        try:
            cmd_ret = run_command(
                ["docker", "manifest", "inspect", remote_name_and_tag],
                stderr=subprocess.PIPE,
            )
            yaml_output = yaml.safe_load(cmd_ret.replace("\t", ""))
            return "schemaVersion" in yaml_output
        except subprocess.CalledProcessError as e:
            error_message = e.stderr.decode("utf-8")
            if f"no such manifest: {remote_name_and_tag}" in error_message:
                return False
            else:
                print("[ERROR] Subprocess error")
                print(error_message)
                raise


def ask_yes(question: str) -> bool:
    return input(f"{question} [y/N]").lower() in ["y", "yes"]


def exit_if_image_exists(
    docker_image_name: str,
    docker_image_tag: str,
    ecr: str,
    message: str = "Docker image '{}:{}' already exists{}",
):
    if (
        image_exists_local(docker_image_name, docker_image_tag)
        or image_exists_registry(docker_image_name, docker_image_tag, ecr)
    ) and not ask_yes(message.format(docker_image_name, docker_image_tag, ", overwrite?")):
        raise RuntimeError(message.format(docker_image_name, docker_image_tag, ". Build aborted!"))


def build_docker_image(
    path: Path,
    dockerfile_path: click.Path,
    credentials_file_path: str,
    pip_credentials_file_path: Optional[str],
    poetry_credentials_file_path: Optional[str],
    uv_credentials_file_path: Optional[str],
    docker_image_name: str,
    docker_image_tags: List[str],
    build_args: List[str],
    docker_target: Optional[DockerTargets],
    build_cache: bool,
    ecr: str,
    upload: bool,
    buildx_builder_name: Optional[str] = None,
    s3_cache_config: Optional[S3CacheConfig] = None,
):
    """Runs the docker build command.

    Args:
        path: Docker context Path.
        dockerfile_path: Path to the dockerfile (relative to this script).
        credentials_file_path: Path to the credentials file.
        docker_image_name: Name of the docker image.
        docker_image_tags: Tags of the docker image.
        build_args: Build args of the docker image.
        docker_target: Target of the docker image.
        build_cache: Whether to use the Docker build cache or not.
        buildx_builder_name: Name of the buildx builder to use for building the image.
            Only required if using S3 cache. Defaults to None.
        s3_cache_config: S3 cache configuration. Defaults to None.
    """
    os.environ["DOCKER_BUILDKIT"] = "1"
    tags = [f"{ecr}/{docker_image_name}:{tag}" for tag in docker_image_tags]
    cmd = ["docker", "build", "-f", dockerfile_path, "--progress", "plain"]
    if not build_cache:
        cmd += ["--no-cache"]

    cmd += ["--secret", f"id=aws_creds,src={credentials_file_path}"]
    if pip_credentials_file_path is not None:
        cmd += ["--secret", f"id=pip_creds,src={pip_credentials_file_path}"]
    if poetry_credentials_file_path is not None:
        cmd += ["--secret", f"id=poetry_creds,src={poetry_credentials_file_path}"]
    if uv_credentials_file_path is not None:
        cmd += ["--secret", f"id=uv_creds,src={uv_credentials_file_path}"]
    if docker_target is not None:
        cmd += ["--target", docker_target.value]

    for arg in build_args:
        cmd += ["--build-arg"] + [arg]

    for tag in tags:
        cmd += [
            "--tag",
            tag,
        ]

    if upload:
        if s3_cache_config is not None:
            cmd += [
                "--builder",
                buildx_builder_name,
                "--cache-to",
                str(s3_cache_config),
                "--cache-from",
                str(s3_cache_config),
            ]
        cmd += ["--push"]

    cmd += [f"{path}"]
    run_command(cmd, verbose=True)


@click.command()
@click.option(
    "--pip-package-name",
    required=True,
    help="Name of the package as installable by pip (e.g. synthesia-{package}).",
)
@click.option(
    "--pip-package-version",
    default=None,
    help="Version of the package. Defaults to latest",
)
@click.option(
    "--python-package-name",
    required=True,
    help="Name of the package as importable in python (e.g. {package})",
)
@click.option(
    "--dockerfile-path",
    default=SCRIPT_DIR / "Dockerfile",
    help=(
        "Path to the Dockerfile that will be used (relative to this script). "
        "Defaults to the Dockerfile within the same folder as the script."
    ),
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--docker-image-name", required=True, help="Name of Docker image")
@click.option(
    "--docker-image-tags",
    default="latest",
    help="Comma-separated string of docker tags, e.g. `{tag_1}, {tag_2}`",
)
@click.option(
    "--extra-build-args",
    default="",
    help="Comma-separated string of build args. Must be of format `{ARG1}={VALUE1}`, {ARG2}={VALUE2}",
)
@click.option(
    "--credentials-file-path",
    default=f"{os.environ['HOME']}/.aws/credentials",
    help="Path to the file containing the aws credentials.",
)
@click.option(
    "--pip-credentials-file-path",
    default=f"{os.environ['HOME']}/.config/pip/pip.conf",
    help="Path to the file containing the pip credentials.",
)
@click.option(
    "--poetry-credentials-file-path",
    default=None,
    help="Path to the file containing the poetry credentials.",
)
@click.option(
    "--uv-credentials-file-path",
    default=None,
    help="Path to the file containing the uv credentials.",
)
@click.option("--upload", is_flag=True, help="Whether to upload to ECR.")
@click.option(
    "--docker-target",
    type=click.Choice(DockerTargets),  # type: ignore
    required=False,
    help="Which docker image target image to build.",
)
@click.option("--aws-account-id", default="812634467557", help="AWS account ID for ECR")
@click.option("--aws-region", default="eu-west-1", help="AWS region for ECR")
@click.option("--disable-cache", is_flag=True, help="Disable docker build cache")
@click.option(
    "--disable-ecr-creation",
    is_flag=True,
    help="Whether to disable the creation of the ECR repo.",
)
@click.option(
    "--use-temp-sso-credentials",
    is_flag=True,
    help="Whether to generate temp SSO credentials for AWS.",
)
@click.option(
    "--overwritable-tags",
    default="latest,dev,stable",
    help="Comma-separated string of Docker image tags can be overwritten. Must be of format `{tag1}, {tag2}`",
)
@click.option(
    "--buildx-builder-name",
    required=False,
    help="Name of the buildx builder to use for building the image. Only required if using S3 cache.",
)
@click.option(
    "--s3-cache-bucket",
    required=False,
    help="S3 bucket to use for caching docker layers. Not specifying it implies no usage of the S3 cache.",
)
def build_image(
    pip_package_name: str,
    pip_package_version: str,
    python_package_name: str,
    dockerfile_path: click.Path,
    docker_image_name: str,
    docker_image_tags: str,
    extra_build_args: str,
    credentials_file_path: str,
    pip_credentials_file_path: str,
    poetry_credentials_file_path: Optional[str],
    uv_credentials_file_path: Optional[str],
    upload: bool,
    docker_target: Optional[DockerTargets],
    aws_account_id: str,
    aws_region: str,
    disable_cache: bool,
    use_temp_sso_credentials: bool,
    disable_ecr_creation: bool,
    overwritable_tags: str,
    buildx_builder_name: Optional[str],
    s3_cache_bucket: Optional[str],
):
    if not os.path.exists(pip_credentials_file_path):
        raise ValueError(
            f"Could not find pip credentials file path at {pip_credentials_file_path}. "
            f"Did you authenticate to CodeArtifact?"
        )

    if disable_cache and s3_cache_bucket is not None:
        raise ValueError("Cannot disable cache but specify an S3 cache at the same time!")

    if s3_cache_bucket is not None and buildx_builder_name is None:
        raise ValueError("S3 cache requires `buildx-builder-name` to be specified!")

    # Create the ECR repository, if config yaml is present
    if not disable_ecr_creation:
        create_ecr_repository()

    if use_temp_sso_credentials is True:
        tmp_file = tempfile.NamedTemporaryFile("w")
        temp_credentials_path = Path(tmp_file.name).resolve()
        generate_temp_sso_credentials(temp_credentials_path)

    build_args = [
        f"PIP_PACKAGE_NAME={pip_package_name}",
        f"PYTHON_PACKAGE_NAME={python_package_name}",
        f"BUILDKIT_INLINE_CACHE={'0' if disable_cache else '1'}",
    ]

    ecr = f"{aws_account_id}.dkr.ecr.{aws_region}.amazonaws.com"

    s3_cache_config = None
    if s3_cache_bucket is not None:
        s3_cache_config = S3CacheConfig(
            bucket=s3_cache_bucket,
            ecr_url=ecr,
            region=aws_region,
            docker_image_name=docker_image_name,
        )

    if pip_package_version is not None:
        build_args.append(f"PIP_PACKAGE_VERSION={pip_package_version}")

    if extra_build_args:
        build_args += parse_build_args_from_str(extra_build_args)

    if docker_image_tags:
        docker_image_tags_list = parse_docker_tags_from_str(docker_image_tags)
    else:
        docker_image_tags_list = ["latest"]

    parsed_overwritable_tags = parse_docker_tags_from_str(overwritable_tags)
    if upload:
        for tag in docker_image_tags_list:
            if tag not in parsed_overwritable_tags:
                exit_if_image_exists(docker_image_name=docker_image_name, docker_image_tag=tag, ecr=ecr)

    with DockerRegistryAccess(ecr), TemporaryDirectory() as context_dir:
        context_dir = Path(context_dir)
        context_dir.mkdir(parents=True, exist_ok=True)
        prepare_artifacts(context_dir, docker_target)
        build_docker_image(
            path=context_dir,
            dockerfile_path=dockerfile_path,
            credentials_file_path=credentials_file_path,
            pip_credentials_file_path=pip_credentials_file_path,
            poetry_credentials_file_path=poetry_credentials_file_path,
            uv_credentials_file_path=uv_credentials_file_path,
            docker_image_name=docker_image_name,
            docker_image_tags=docker_image_tags_list,
            build_args=build_args,
            docker_target=docker_target,
            build_cache=not disable_cache,
            buildx_builder_name=buildx_builder_name,
            s3_cache_config=s3_cache_config,
            ecr=ecr,
            upload=upload,
        )


if __name__ == "__main__":
    build_image()  # type: ignore
