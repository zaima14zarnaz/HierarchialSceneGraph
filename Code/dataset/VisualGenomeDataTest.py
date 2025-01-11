import json
import cv2

class Obj:
    def __init__(self, name, width, height, x, y, objID, attributes):
        self.name = name
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.objID = objID
        self.attributes = attributes



with open('/Users/zaimazarnaz/PycharmProjects/HierarchicalSceneGraph/Dataset/visualGenome/attributes.json') as f:
    attr_dataset = json.load(f)


# Access the first image metadata
object_data = attr_dataset[1]
image_id = object_data['image_id']
objects = object_data['attributes']
obj_list = []


img = cv2.imread('/Users/zaimazarnaz/PycharmProjects/HierarchicalSceneGraph/Dataset/visualGenome/VG_100K/2.jpg')
for obj_data in objects:

    new_obj = Obj(obj_data['names'], obj_data['w'], obj_data['h'], obj_data['x'], obj_data['y'], obj_data['object_id'], None)


    top_left = (new_obj.x, new_obj.y)
    bottom_right = (new_obj.x + new_obj.width, new_obj.y + new_obj.height)

    cv2.rectangle(img, top_left, bottom_right, color=(0,0,255), thickness=2)

    if 'attributes' in obj_data:
        new_obj.attributes = obj_data['attributes']
        label = new_obj.name[0] + "(" + ", ".join(new_obj.attributes) + ")"
    else:
        label = new_obj.name[0]
    label_pos = (new_obj.x, new_obj.y - 10)
    cv2.putText(img, label, label_pos, color=(0,0,0), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.0, thickness=1)

    obj_list.append(new_obj)

cv2.imshow('image_1', img)
cv2.waitKey(0)
cv2.destroyAllWindows()



