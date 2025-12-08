"""
Forecaster Factory - Fábrica unificada para criação de forecasters

Centraliza a criação de todos os tipos de forecasters (Load, Solar).
"""

from typing import Dict, Optional
from .load_forecaster import LoadForecaster
from .solar_forecaster import SolarForecaster
from .profile_persistence_forecaster import ProfilePersistenceForecaster


class ForecastFactory:
    """
    Fábrica para criar forecasters de forma unificada.

    Esta classe centraliza a criação de forecasters, tornando mais fácil
    adicionar novos métodos de previsão e manter consistência na configuração.
    """

    @staticmethod
    def create_load_forecaster(config: Dict, house, method: Optional[str] = None) -> LoadForecaster:
        """
        Cria um LoadForecaster a partir da configuração.

        Args:
            config: Dicionário de configuração completo
            house: Instância da classe House com dados históricos
            method: Método de previsão a usar (override da config)
                   Opções: 'mean', 'naive', 'moving_average', 'historical',
                          'profile_persistence', 'deep_autoformer'

        Returns:
            LoadForecaster configurado

        Example:
            >>> factory = ForecastFactory()
            >>> load_forecaster = factory.create_load_forecaster(config, house)
            >>> # Ou com override do método:
            >>> load_forecaster = factory.create_load_forecaster(config, house, method='deep_autoformer')
        """
        # Get load forecaster config if it exists, otherwise use defaults
        forecaster_config = config.get('load_forecaster', {})

        # Deep Autoformer config (usado apenas se method='deep_autoformer')
        deep_config = forecaster_config.get('deep_autoformer', {})
        deep_autoformer_config = {
            'base_path': deep_config.get('base_path'),
            'model_path': deep_config.get('model_path'),
            'config_path': deep_config.get('config_path'),
            'data_path': deep_config.get('data_path')
        } if deep_config else None

        # Profile persistence config (usado apenas se method='profile_persistence')
        profile_config = forecaster_config.get('profile_persistence', {})
        profile_persistence_config = {
            'n_weeks': profile_config.get('n_weeks', 3)
        }

        # Create load forecaster com todas as configs disponíveis
        load_forecaster = LoadForecaster(
            house=house,
            deep_autoformer_config=deep_autoformer_config,
            profile_persistence_config=profile_persistence_config
        )

        return load_forecaster

    @staticmethod
    def create_solar_forecaster(config: Dict, solar_panel, method: Optional[str] = None) -> SolarForecaster:
        """
        Cria um SolarForecaster a partir da configuração.

        Args:
            config: Dicionário de configuração completo
            solar_panel: Instância da classe SolarPanel com dados históricos
            method: Método de previsão a usar (override da config)
                   Opções: 'mean', 'naive', 'moving_average', 'historical'

        Returns:
            SolarForecaster configurado

        Example:
            >>> factory = ForecastFactory()
            >>> solar_forecaster = factory.create_solar_forecaster(config, solar_panel)
        """
        # Por agora, SolarForecaster não usa Deep Autoformer
        # Apenas métodos baseline
        solar_forecaster = SolarForecaster(solar_panel=solar_panel)

        return solar_forecaster

    @staticmethod
    def create_profile_persistence_forecaster(config: Dict,
                                               historical_data,
                                               value_column: str,
                                               dt_minutes: int = 15) -> ProfilePersistenceForecaster:
        """
        Cria um ProfilePersistenceForecaster a partir da configuração.

        Usado principalmente pelo SDP controller para previsões rápidas.

        Args:
            config: Dicionário de configuração completo
            historical_data: DataFrame com dados históricos (timestamp, value)
            value_column: Nome da coluna com valores a prever
            dt_minutes: Intervalo de tempo em minutos (default=15)

        Returns:
            ProfilePersistenceForecaster configurado

        Example:
            >>> factory = ForecastFactory()
            >>> forecaster = factory.create_profile_persistence_forecaster(
            ...     config, historical_data, value_column='consumption'
            ... )
        """
        # Get config for profile persistence
        sdp_config = config.get('controller', {}).get('sdp', {})
        forecaster_config = sdp_config.get('forecaster', {})
        n_weeks = forecaster_config.get('n_weeks', 3)

        # Create forecaster
        forecaster = ProfilePersistenceForecaster(
            n_weeks=n_weeks,
            dt_minutes=dt_minutes
        )

        # Set historical data
        forecaster.set_data(historical_data, value_column=value_column)

        return forecaster

    @staticmethod
    def validate_forecaster_method(method: str, forecaster_type: str = 'load') -> bool:
        """
        Valida se um método de previsão é válido para o tipo de forecaster.

        Args:
            method: Nome do método a validar
            forecaster_type: Tipo de forecaster ('load' ou 'solar')

        Returns:
            True se o método é válido, False caso contrário

        Example:
            >>> ForecastFactory.validate_forecaster_method('deep_autoformer', 'load')
            True
            >>> ForecastFactory.validate_forecaster_method('deep_autoformer', 'solar')
            False
        """
        valid_methods = {
            'load': ['mean', 'naive', 'moving_average', 'historical',
                    'profile_persistence', 'deep_autoformer'],
            'solar': ['mean', 'naive', 'moving_average', 'historical']
        }

        return method in valid_methods.get(forecaster_type, [])

    @staticmethod
    def get_available_methods(forecaster_type: str = 'load') -> list:
        """
        Retorna lista de métodos disponíveis para um tipo de forecaster.

        Args:
            forecaster_type: Tipo de forecaster ('load' ou 'solar')

        Returns:
            Lista de métodos disponíveis

        Example:
            >>> ForecastFactory.get_available_methods('load')
            ['mean', 'naive', 'moving_average', 'historical', 'profile_persistence', 'deep_autoformer']
        """
        available_methods = {
            'load': ['mean', 'naive', 'moving_average', 'historical',
                    'profile_persistence', 'deep_autoformer'],
            'solar': ['mean', 'naive', 'moving_average', 'historical']
        }

        return available_methods.get(forecaster_type, [])
