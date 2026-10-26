# eflips.eval conventions

- `input/` must not touch `Event` or `Vehicle`. Anything that needs simulation results goes in `output/`.
- `prepare.foo` and `visualize.foo` come in pairs with the same name. Variants such as `visualize.foo_hist` are
  allowed. A new pair needs a test in `tests/test_input.py` or `tests/test_output.py`.
- `prepare.foo(id_or_ids, session)` does all database access and returns a DataFrame. The depot functions are
  existing exceptions.
- `visualize.foo(df, **kwargs)` sets axis labels and a legend. It never sets a title or a figure size.
- Labels are English and include units, e.g. `"Distance [km]"`, not `total_dist`.
