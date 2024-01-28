Render notebooks with

```bash
python -m ipykernel install --name kernel_name --user
scripts/build.sh src/notebooks/[^_]*.py
```

Edit notebooks interactively with

```bash
jupyter notebook --notebook-dir src/notebooks
```
