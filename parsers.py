from icevision.all import *
# from icevision.imports import *
import pandas as pd


__all__ = ['PlainTxtParser']


class PlainTxtParser(Parser):
    def __init__(self, img_dir, txt_dir, class_map: ClassMap):
        txt_files = get_files(txt_dir, extensions='.txt', recurse=False)
        self.id2path = {file.stem: file for file in get_image_files(img_dir)}
        self.inner_df = pd.DataFrame(self.parse_txts(txt_files), columns=['id', 'txt', 'bbox', 'label'])
        self.class_map = class_map
        super().__init__(self.template_record())

    def parse_fields(self, o, record, is_new: bool):
        if is_new:
            record.set_filepath(self.id2path[o.id])
            record.set_img_size(get_img_size(record.filepath))
            record.detection.set_class_map(self.class_map)
        bbox = BBox.from_xyxy(*o.bbox)
        record.detection.add_bboxes([bbox])
        record.detection.add_areas([bbox.area])
        record.detection.add_labels([o.label])

    def get_txt_annotations(self, txt_file_path):
        bboxes, labels = [], []
        try:
            with open(txt_file_path) as text_file:
                data = text_file.readlines()

            for line in data:
                bbox, label = line.strip().split('|')
                bbox = [float(px) for px in bbox.split(',')]
                bboxes.append(bbox)
                labels.append(label)
        except FileNotFoundError:
            warnings.warn(f"{txt_file_path} not found, inserting dummy labels")
            bboxes.append([0, 0, 0, 0])
            labels.append(0)
        return bboxes, labels

    def __iter__(self):
        yield from self.inner_df.itertuples()

    def record_id(self, o):
        return o.id

    def parse_txts(self, txt_files):
        for i, file in enumerate(txt_files):
            bboxes, labels = self.get_txt_annotations(file)
            for bbox, label in zip(bboxes, labels):
                yield file.stem, file.name, bbox, label

    @staticmethod
    def template_record() -> BaseRecord:
        return BaseRecord(
            (
                FilepathRecordComponent(),
                InstancesLabelsRecordComponent(),
                BBoxesRecordComponent(),
                AreasRecordComponent(),
            )
        )
