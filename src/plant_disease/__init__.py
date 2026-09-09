"""Plant disease classification on the strata catalog.

Two plugins, registered through strata's own entry point groups:

- :mod:`plant_disease.preparer` reads PlantVillage's directory layout as
  candidate labels and leaf groups.
- :mod:`plant_disease.model` trains and predicts over a materialised
  dataset version.

Nothing here imports the labeller or a database. The preparer sees files,
the model sees paths and values, and neither knows a catalog exists.
"""
