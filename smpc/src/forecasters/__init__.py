"""
Forecasters package - Módulos de previsão para carga e produção fotovoltaica

Este pacote contém forecasters baseados em métodos simples (baselines)
inspirados no deep_autoformer_minimal.
"""

from .load_forecaster import LoadForecaster
from .solar_forecaster import SolarForecaster
from .profile_persistence_forecaster import ProfilePersistenceForecaster

__all__ = ['LoadForecaster', 'SolarForecaster', 'ProfilePersistenceForecaster']
