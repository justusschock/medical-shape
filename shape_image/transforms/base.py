from typing import Sequence
from rising.transforms import AbstractTransform
from shape_image.image import ShapeImage, BatchedShapeImage
from abc import ABCMeta, abstractmethod


class  BaseShapeImageTransform(AbstractTransform, metaclass=ABCMeta):
    """
    Provides the general toolset to apply given transformations on shapeimages by using a transformation Class/Object
    """

    def __init__(
        self,
        shapeimage_query_keys: Sequence = ("data",),
        grad: bool = False,
        **kwargs
    ):
        """
        Args:
            shapeimage_query_keys:
                The keys that provide a ShapeImage in assembled(ShapeImage object) or not assembled(as a dict) form
            grad:
                Whether or not to calculate the gradient
            
        """
        super().__init__(grad=grad, **kwargs)

        self.shapeimage_query_keys = shapeimage_query_keys

    @abstractmethod
    def apply_transform(self, shape_image: ShapeImage):
        raise NotImplementedError

    def assemble_if_necessary(self, data: dict):
        """
        Will convert every dicts to shapeimages that are reverenced in self.keys
        Args:
            data:
        Returns:
            New data dict and a dict that says witch key had to be assemble
        """
        assembly_necessary_dict = {}
        for key in self.shapeimage_query_keys:
            val = data[key]
            if isinstance(val, ShapeImage):
                assembly_necessary_dict[key] = False
                continue

            elif isinstance(val, dict) and ("shape" in val or "image" in val):
                shape = val.get("shape", None)
                image = val.get("image", None)
                data[key] = BatchedShapeImage(image=image, shape=shape)

                assembly_necessary_dict[key] = True

            else:
                raise ValueError
        return data, assembly_necessary_dict

    def dissemble_if_specified(self, data: dict, assembly_necessary: dict):
        """
        Will convert every shapeimages into dicts that are reverenced in assembly_necessary and self.shapeimage_query_keys
        This function should reverse the assemble_if_necessary function
        Args:
            data:
            assembly_necessary:
                Dict with a key:bool combination that says which shapeimages has to be converted to dicts so that the
                data dict has the same structure as before the conversion to shapeimages
        Returns:
            Datadict with (some) dissembled shapeimages
        """
        for key in self.shapeimage_query_keys:
            if assembly_necessary[key]:
                data[key] = {"image": data[key].image, "shape": data[key].shape}

        return data

    def pre_calc(self, key: str, val: ShapeImage):
        """
        Calculates parameters and values that are used by the transformation function
        Args:
            key:
                The name of the data dict entry
            val:
                The datadict entry.
                This is a shapeimage
        """
        pass

    def forward(self, **data) -> dict:
        """
        Calls the transformation function or applies the transformation for the shapeimages in the data dict.
        The transformation is only applied in the shapeimages revered in self.shapeimage_query_keys
        Args:
            **data:
                The data dict with the shapeimages
        Returns:
            Data dict with transformed shapeimages
        """
        assembled_data, assembly_necessary = self.assemble_if_necessary(data)

        for key in self.shapeimage_query_keys:
            val = assembled_data[key]
            self.pre_calc(key=key, val=val)
            assembled_data[key] = self.apply_transform(val)

        return self.dissemble_if_specified(assembled_data, assembly_necessary)

class ShapeImageFunctionTransform(BaseShapeImageTransform):
    def __init__(self, func_name, shapeimage_query_keys: Sequence[str] = ('data',), **kwargs):
        super().__init__(shapeimage_query_keys=shapeimage_query_keys)
        self.func_kwargs = kwargs
        self.func_name = func_name

    def apply_transform(self, shape_image: ShapeImage):
        return getattr(shape_image, self.func_name)(**self.func_kwargs)

