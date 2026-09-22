# Project Guidance

- This directory is the finished September 22 app for comparison with student builds. The exercise starts from the separate in-class-notebooks/TSP project.
- Keep the example focused on facility coordinates, tour visualization, and tour distance. Distribution-center selection and vehicle-routing constraints are out of scope.
- Run Python, marimo, and tests through pixi.
- Numeric results do not need LaTeX-importable exports.
- Treat the Parquet files under `data/` as class-ready inputs. Never overwrite them in place. The one exception on record is the September 2026 restoration of the `city`, `state`, and `zip` fields to `store_locations.parquet`, which the owner approved explicitly. The rule stands for every other change.
