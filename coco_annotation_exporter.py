from icevision.imports import *
from icevision import ClassMap, BBox, BaseRecord, pbar


@dataclass
class COCOAnnotation:
    images: List[Dict] = dataclasses.field(default_factory=list)
    annotations: List[Dict] = dataclasses.field(default_factory=list)
    categories: List[Dict] = dataclasses.field(default_factory=list)


class COCOAnnotationExporter:
    def __init__(self, class_map: ClassMap, img_dir: Optional[Union[str, Path]] = None):
        self.__annotation_id = 0
        self.class_map = class_map
        self.img_dir = img_dir

    def get_annotation_id(self):
        # increasing first is fine, as labels are expected to be [1..N]
        self.__annotation_id += 1
        return self.__annotation_id

    def export(self, outfile='coco_formatted_annotations.json', records: Collection[BaseRecord] = tuple()):
        logger.info(
            f"Saving records in coco format to {outfile}",
        )
        coco = COCOAnnotation()
        for record in pbar(records):
            coco.images.extend(self.create_image_annotations(record))
            coco.annotations.extend(self.create_coco_annotations(record))
        coco.categories = self.create_categories()

        with open(outfile, mode='w') as outfile:
            json.dump(dataclasses.asdict(coco), outfile, indent=2)

    def create_single_annotation(self, bbox: BBox, label: str, image_id: int):
        xywh = bbox.xywh
        area = bbox.area
        label_id = self.class_map.get_by_name(label)
        annotation_id = self.get_annotation_id()

        annotation = {
            'id': annotation_id,
            'image_id': image_id,
            'bbox': xywh,
            'area': area,
            'iscrowd': 0,
            'category_id': label_id,
            'segmentation': []
        }

        return annotation

    def create_coco_annotations(self, record):
        image_id = record.record_id
        annotations = [
            self.create_single_annotation(bbox, label, image_id)
            for bbox, label in zip(record.detection.bboxes, record.detection.labels)
        ]

        return annotations

    def create_image_annotations(self, record):
        file_path = record.filepath.relative_to(self.img_dir).as_posix()
        width = record.width
        height = record.height
        image_id = record.record_id
        images = [{
            'file_name': file_path,
            'height': height,
            'width': width,
            'id': image_id
        }]

        return images

    def create_categories(self):
        categories = []
        for label, idx in self.class_map._class2id.items():
            category = {
                "supercategory": label,
                "id": idx,
                "name": label
            }
            categories.append(category)
        return categories