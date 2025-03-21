# Copyright (c) 2025 Synthesia Limited - All Rights Reserved
#
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.

import os

from setuptools import setup  # type: ignore
from setuptools_scm import ScmVersion
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def resolve_version():
    """This function retrieves the correct version for the package

    The function checks the env variables to retrieve info about the
    state. If we are on PR we will use the default versioning scheme.
    If we are merging on main, then we will have a STABLE RELEASE variable
    set to 1. This avoid to add the local version to the version number.
    """

    def my_release_branch_semver_version(version: ScmVersion):
        if os.getenv("STABLE_RELEASE"):
            return os.getenv("BASE_VERSION")

        if os.getenv("DEV_VERSION"):
            return os.getenv("DEV_VERSION")

        return version.format_with("{tag}.dev0")

    version_dictionary = {"version_scheme": my_release_branch_semver_version}
    if os.getenv("STABLE_RELEASE") or os.getenv("DEV_VERSION"):
        version_dictionary["local_scheme"] = "no-local-version"  # type: ignore

    return version_dictionary


setup(
    ext_modules=[
        CUDAExtension(
            name="gnnhwc._ops",
            sources=["src/gnnhwc/csrc/custom_gn.cpp", "src/gnnhwc/csrc/gn_kernel.cu"],
            extra_compile_args={
                "cxx": [
                    "-Ofast",  # needed or else GN NCHW from source is slower than nn.GroupNorm
                    "-funroll-all-loops",
                    "-march=x86-64-v3",
                    "-mtune=znver3",
                ],
                "nvcc": [
                    "-use_fast_math",
                    "-extra-device-vectorization",
                    "-extended-lambda",  # for gpu_kernel (although this isn't used in custom GN kernels)
                    "-lineinfo",  # useful for profiling
                    "-src-in-ptx",
                ],
            },
            py_limited_api=True,
            no_python_abi_suffix=True,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
    use_scm_version=resolve_version,
)
