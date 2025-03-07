# Copyright (c) 2025 Synthesia Limited - All Rights Reserved
#
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.

import timeit


def test_example():
    # Setup
    a = 5
    desired = 10

    # Exercise
    actual = a * 2

    # Verify
    assert actual == desired

    # Cleanup


def test_performance_example():
    # Setup
    desired = 0.35

    # Exercise
    actual = timeit.timeit('"-".join(str(n) for n in range(100))', number=10000)

    # Verify
    assert actual <= desired

    # Cleanup
