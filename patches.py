from typing import Tuple
from fastcore.foundation import patch
from icevision import parsers, FilepathRecordComponent
from torchvision import io


@patch
def image_width_height(self:parsers.COCOBaseParser, o) -> Tuple[int, int]:
    return self._info['width'], self._info['height']


@patch
def _load(self:FilepathRecordComponent):
    torch_img = io.read_image(str(self.filepath))
    c, self.height, self.width = torch_img.shape
    if c == 1: torch_img = torch_img.float().mean(dim=0).repeat((3,1,1)).byte()
    # CHW to HWC
    img = torch_img.permute(1,2,0).numpy()
    self.set_img(img)
    return

