class FrameworkError(Exception):
    """Error class for handing custom exceptions within the framework.
    """

    @classmethod
    def from_attribute(cls, obj, attribute):
        """Raise a `FrameworkError` from an obj with a misued attribute.
        
        Parameters
        ----------
        `obj`: `object`.
            Whose attributes were used incorrectly.
            
        `attribute`: `str`.
            The name of the attribute.
        """
        return cls(f"{obj.__class__}.{attribute} can not be used in that way.")

    @classmethod
    def from_runtime(cls, message):
        """Raise a `FrameworkError` from a runtime misconfiguration.
        
        Parameters
        ----------
        `message`: `str`.
            To use for the `FrameworkError`.
        """
        return cls(message)

    @classmethod
    def from_value(cls, field, expected, received):
        """Raise a `FrameworkError` from an incorrect value.
        
        Parameters
        ----------
        `field`: `str`.
            Whose values are incorrect.
            
        `expected`: `object`.
            The expected value of `field`.
            
        `received`: `object`.
            The value of `field`.
        """
        return cls(f"Expected {expected} for '{field}', but received: {received}.")

    @classmethod
    def from_type(cls, field, expected, received):
        """Raise a `FrameworkError` from an incorrect type.
        
        Parameters
        ----------
        `field`: `str`.
            Whose type is incorrect.
            
        `expected`: `object`.
            The expected type of `field`.
            
        `received`: `object`.
            The value of `field`.
        
        Note
        ----
        Do not pass in the `type(received)` for received, as this is done within
        the exception.
        """
        return cls(
            f"Expected {expected} for '{field}', but received: {type(received)}."
        )
