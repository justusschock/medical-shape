from torch.utils.data._utils.collate import default_collate

from shape_image.image import ShapeImage

def shape_collate(batch):
    elem = batch[0]
    if isinstance(elem, ShapeImage):
        return sum(batch)
    else:
        return default_collate(batch)
