# defumat has moved

This package is now called **defuqumat**, density functional for quantum materials, and its
repository is <https://github.com/joselado/defuqumat>. The history up to the rename is
there as well, so nothing is lost by cloning the new one; this repository is kept only so
that old links and clones find their way, and the code in it is frozen at the last commit
before the rename.

To move an existing clone over:

```bash
git remote set-url origin git@github.com:joselado/defuqumat.git
git pull
pip uninstall defumat && pip install -e .
```

Note that every `DEFUMAT_*` environment variable is now `DEFUQUMAT_*`, so a job script
that still sets the old name runs without error but with the defaults.
