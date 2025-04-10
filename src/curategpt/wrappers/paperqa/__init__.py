"""PaperQA wrapper module for searching document collections."""

try:
    from .paperqawrapper import PaperQAWrapper
    __all__ = ["PaperQAWrapper"]
except ImportError:
    # Create a dummy class that raises a helpful error message when used
    class PaperQAWrapper:
        name = "paperqa"
        
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Could not import PaperQAWrapper. This could be due to missing dependencies. "
                "To use PaperQAWrapper, install curategpt with 'pip install curategpt[paperqa]' "
                "or install the dependencies directly with 'pip install paper-qa langchain-community'"
            )
    
    __all__ = ["PaperQAWrapper"]
