# Tests

The `session` fixture drops every table in `DATABASE_URL`. Only run tests against a throwaway PostgreSQL database
that has PostGIS and `CREATE EXTENSION btree_gist`. Ask before running them if you don't know what
`DATABASE_URL` points to.

- All: `poetry run pytest`
- One: `poetry run pytest tests/test_output.py::TestOutput::test_vehicle_soc`

`test_output.py` runs the eflips-depot simulation in a module fixture, so it is slow. Never edit a test to make it
pass. If a test looks wrong, say so.
