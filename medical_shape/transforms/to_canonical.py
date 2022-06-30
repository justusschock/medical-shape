from medical_shape.transforms.to_orientation import ToOrientation
from typing import Optional, Union, Tuple, List

class ToRAS(ToOrientation):
    def __init__(self, shape_trafo_image_size: Optional[Union[Tuple[int, ...], List[int]] = None, shape_trafo_image_key: Optional[str] = None, **kwargs):
        super().__init__(axcode=('R', 'A', 'S'), shape_trafo_image_key=shape_trafo_image_key, shape_trafo_image_size=shape_trafo_image_size)

ToCanonical = ToRAS