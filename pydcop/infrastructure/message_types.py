from typing import List

from pydcop.utils.simple_repr import simple_repr, SimpleReprException, SimpleRepr


def message_type(msg_type: str, fields: List[str]):
    """
    Class factory method for Messages

    This utility method can be used to easily define new Message type without
    subclassing explicitly (and manually) the Message class. Tt output a
    class object which subclass Message.

    Message instance can be created from the return class type using either
    keywords arguments or positional arguments (but not both at the same time).

    Instances from Message classes created with `message_type` support
    equality, simple_repr and have a meaningful str representation.

    Parameters
    ----------
    msg_type: str
        The type of the message, this will be return by `msg.type` (see example)
    fields: List[str]
        The fields in the message

    Returns
    -------
    A class type that can be used as a message type.

    Example
    -------

    >>> MyMessage = message_type('MyMessage', ['foo', 'bar'])
    >>> msg1 = MyMessage(foo=42, bar=21)
    >>> msg = MyMessage(42, 21)
    >>> msg.foo
    42
    >>> msg.type
    'MyMessage'
    >>> msg.size
    0
    """

    def __init__(self, *args, **kwargs):
        if args and kwargs:
            raise ValueError("Use positional or keyword arguments, but not " "both")
        if args:
            if len(args) != len(fields):
                raise ValueError("Wrong number of positional arguments")
            for f, a in zip(fields, args):
                setattr(self, f, a)

        for k, v in kwargs.items():
            if k not in fields:
                raise ValueError("Invalid field {k} in {msg_type}")
            setattr(self, k, v)
        Message.__init__(self, msg_type, None)

    def to_str(self):
        fs = ", ".join([f + ": " + str(getattr(self, f)) for f in fields])
        return msg_type + "(" + fs + ")"

    def _simple_repr(self):

        # Full name = module + qualifiedname (for inner classes)
        r = {
            "__module__": self.__module__,
            "__qualname__": "message_type",
            "__type__": self.__class__.__qualname__,
        }
        for arg in fields:
            try:
                val = getattr(self, arg)
                r[arg] = simple_repr(val)
            except AttributeError:
                if hasattr(self, "_repr_mapping") and arg in self._repr_mapping:
                    try:
                        r[arg] = self.__getattribute__(self._repr_mapping[arg])
                    except AttributeError:
                        SimpleReprException(
                            f"Invalid repr_mapping in {self}, "
                            "no attribute for {self._repr_mapping[arg]}"
                        )

                else:
                    raise SimpleReprException(
                        "Could not build repr for {self}, " "no attribute for {arg}"
                    )
        return r

    def equals(self, other):
        if self.type != other.type:
            return False
        if self.__dict__ != other.__dict__:
            return False
        return True

    msg_class = type(
        msg_type,
        (Message,),
        {
            "__init__": __init__,
            "__str__": to_str,
            "__repr__": to_str,
            "_simple_repr": _simple_repr,
            "__eq__": equals,
        },
    )
    return msg_class


class Message(SimpleRepr):
    """
    Base class for messages.

    you generally sub-class ``Message`` to define the message type for a DCOP
    algorithm.
    Alternatively you can use :py:func:`message_type` to create
    your own message type.


    Parameters
    ----------
    msg_type: str
       the message type ; this will be used to select the correct handler
       for a message in a DcopComputation instance.
    content: Any
       optional, usually you sub-class Message and add your own content
       attributes.

    """

    def __init__(self, msg_type, content=None):
        self._msg_type = msg_type
        self._content = content

    @property
    def size(self):
        """
        Returns the size of the message.

        You should overwrite this methods in subclasses,
        will be used when computing the communication load of an
        algorithm and by some distribution methods that optimize
        the distribution of computation for communication load.

        Returns
        -------
        size : int

        """
        return 0

    @property
    def type(self) -> str:
        """
        The type of the message.

        Returns
        -------
        message_type: str
        """
        return self._msg_type

    @property
    def content(self):
        return self._content

    def __str__(self):
        return f"Message({self.type})"

    def __repr__(self):
        return f"Message({self.type}, {self.content})"

    def __eq__(self, other):
        if type(other) != type(self):
            return False
        return self.type == other.type and self.content == other.content


ConstraintEvaluationRequest = message_type(
    'constraint_evaluation_request', ['constraint_name', 'var_assignments'],
)

ConstraintEvaluationResponse = message_type(
    'constraint_evaluation_response', ['constraint_name', 'value'],
)

AgentMovedMessage = message_type(
    'agent_moved', ['prev_position', 'new_position']
)

# SimTimeStepChange is sent by the orchestrator to all agents when the simulation environment changes its times step
SimTimeStepChanged = message_type(
    'sim_time_step_change', ['data'],
)

# Sent from a stabilization computation to DCOP computation(s)
DcopExecutionMessage = message_type(
    'dcop_execution_message', [],
)

DcopConfigurationMessage = message_type(
    'dcop_configuration_message', ['data'],
)

DcopInitializationMessage = message_type(
    'dcop_initialization_message', [],
)

ASYNC_MSG_TYPES = [
    'constraint_evaluation_request',
    'constraint_evaluation_response',
]
