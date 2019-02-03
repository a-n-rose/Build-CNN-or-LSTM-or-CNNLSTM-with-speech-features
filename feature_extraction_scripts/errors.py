
class Error(Exception):
    """Base class for other exceptions"""
    pass

class DatasetMixing(Error):
    pass

class NoSpeechDetected(Error):
    pass

class LimitTooSmall(Error):
    pass

class FeatureExtractionFail(Error):
    pass

class ExitApp(Error):
    pass
