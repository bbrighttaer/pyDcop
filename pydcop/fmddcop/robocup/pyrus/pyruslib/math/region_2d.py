"""
  \ file region_2d.h
  \ brief abstract 2D region class File.

"""

from pydcop.fmddcop.robocup.pyrus.pyruslib.math.vector_2d import Vector2D


class Region2D:
    """
      \ brief accessible only from pydcop.fmddcop.robocup.pyrus.derived classes
    """

    def __init__(self):
        pass

    """
      \ brief get the area of this region
      \ return value of the area
    """

    def area(self):
        pass

    """
      \ brief check if this region contains 'point'.
      \ param point considered point
      \ return true or false
    """

    def contains(self, point: Vector2D):
        pass