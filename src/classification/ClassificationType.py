from enum import Enum


class ClassificationType(str, Enum):
    BINARY = 'binary',
    MULTICLASS = 'multiclass'

    @staticmethod
    def get_type(name: str) -> 'ClassificationType':
        classification_type = ClassificationType.MULTICLASS
        for ct in ClassificationType:
            if ct == name:
                classification_type = ct
                break

        return classification_type
