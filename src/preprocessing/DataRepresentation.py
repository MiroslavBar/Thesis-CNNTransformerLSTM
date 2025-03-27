from enum import Enum


class DataRepresentation(str, Enum):
    TIME_SERIES = 'time_series',
    FREQUENCY = 'frequency',
    TIME_FREQUENCY = 'time_frequency',
    OTHER = 'other'

    @staticmethod
    def get_representation(name: str) -> 'DataRepresentation':
        representation = DataRepresentation.TIME_SERIES
        for dr in DataRepresentation:
            if dr == name:
                representation = dr
                break

        return representation
