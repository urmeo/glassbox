"""Glass Box — measuring whether a generated interface conveys understanding.

The package is organized as a pipeline:

    scenario (data) -> interface (render) -> stimulus -> reader -> answer -> score

with a preference judge running alongside the readers, an H1 analysis comparing
what is *preferred* against what is *understood*, and an H4 repair loop that
regenerates an interface to fix a reader's specific mistake.

The core imports only the standard library.
"""

__version__ = "0.1.0"
