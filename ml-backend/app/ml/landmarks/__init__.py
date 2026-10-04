"""ESGE 10-station landmark recognition (schema esge10.v1).

Self-contained feature package: taxonomy, screen layouts and preprocessing
shared by training and serving, frame quality, and the live station tracker.
Nothing here touches the PEACE model or its preprocessing.

Modules that need torch import it lazily, so the mock path runs without it.
"""
