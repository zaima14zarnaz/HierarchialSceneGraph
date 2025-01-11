import os
import json
from pathlib import Path

class DataPrep:
    label_map = {}
    def __init__(self):
        self.annotations = []

    def parse_dataset(self, object_info_path, img_path, filename):
        print("Parsing dataset...")

        # Path to the dataset
        dataset_dir = Path(object_info_path)
        objects_path = dataset_dir / filename

        label_idx = 0


        # Parse objects.json
        with open(objects_path, 'r') as f:
            objects_data = json.load(f)

        # Example: Extract image ID, object labels, and bounding boxes
        for img_data in objects_data:
            img_id = img_data['image_id']
            img_loc = os.path.join(img_path, "{}.jpg".format(img_id))

            objects = {}
            if not os.path.exists(img_loc):
                continue
            for obj in img_data['objects']:
                if obj['w'] == 0:
                    obj['w'] = 1
                if obj['h'] == 0:
                    obj['h'] = 1
                bbox = [obj['x'], obj['y'], obj['x'] + obj['w'], obj['y'] + obj['h']]
                label = obj['names'][0]  # Use the first name as the label
                if label not in self.label_map:
                    DataPrep.label_map[label] = label_idx
                    label_idx += 1
                objects[DataPrep.label_map[label]] = bbox

            if len(objects) == 0:
                continue

            annotations = {'image_id': img_id, 'label': list(objects.keys()), 'bbox': list(objects.values())}
            print(annotations)

            self.annotations.append(annotations)



        return self.annotations, self.label_map, label_idx


