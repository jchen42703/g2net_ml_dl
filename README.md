# g2net_ml_dl

# Workflows

## Installing the library

```
cd python
pip install .
```

Then you can import it into whatever package you need. This makes it convenient to use outside of just this repo.

- Downside: If you need to make changes to the library, you must reinstall the package for the new changes to work

## Full Workflow

```
git clone -b YOUR_BRANCH_NAME https://github.com/jchen42703/g2net_ml_dl
cd g2net_ml_dl/python
pip install .

# Then run your scripts from here or just import stuff from the library ^^.
```

If you're debugging a lot, I recommend:

```
import os
os.chdir("python")
import g2net
os.chdir("..")
os.getcwd()
```

A simple kernel restart will update your changes, so you won't need to reinstall the library with `pip install .`
