import os
import cv2
import copy
import json
import tqdm
import numpy as np

from data.cityscapes import is_label, name2index


def fill_polygon(polygon):
    min_point = polygon.min(0)
    max_point = polygon.max(0)
    box_size = max_point-min_point
    img = np.zeros((box_size[1], box_size[0]), dtype=np.uint8)
    img = cv2.fillPoly(img, [polygon-min_point], 1)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours[0].reshape(-1, 2)+min_point


if __name__ == "__main__":
    root = 'D:\cityscapes'
    subset = 'train'
    labels_root = os.path.join(root, 'gtFine/' + subset)

    filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(labels_root)) for f in
                        fn if is_label(f)]
    filenamesGt.sort()

    for filename in tqdm.tqdm(filenamesGt):
        with open(filename, 'r') as f:
            label_json = json.load(f)

        base_name = os.path.basename(filename)
        n_label_json = copy.deepcopy(label_json)
        n_label_json["objects"].clear()
        for obj in label_json["objects"]:
            # handle category id
            label_name = obj["label"]
            if label_name not in name2index:
                continue
            # handle boundary
            polygon = fill_polygon(np.array(obj["polygon"], dtype=np.int32))
            n_label_json["objects"].append({"label": label_name, "polygon": polygon.tolist()})

        with open(filename.replace("polygons.json", "fill_polygons.json"), 'w') as f:
            json.dump(n_label_json, f, indent=1)




