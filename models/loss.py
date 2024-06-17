from tensorflow import stack, maximum, minimum, clip_by_value, concat
import tensorflow as tf
import numpy as np
from settings import *

def ComputeIoU(boxes1, boxes2):
    boxes1_t = stack([boxes1[..., 0] - boxes1[..., 2] / 2.0,
                         boxes1[..., 1] - boxes1[..., 3] / 2.0,
                         boxes1[..., 0] + boxes1[..., 2] / 2.0,
                         boxes1[..., 1] + boxes1[..., 3] / 2.0],
                        axis=-1)

    boxes2_t = stack([boxes2[..., 0] - boxes2[..., 2] / 2.0,
                         boxes2[..., 1] - boxes2[..., 3] / 2.0,
                         boxes2[..., 0] + boxes2[..., 2] / 2.0,
                         boxes2[..., 1] + boxes2[..., 3] / 2.0],
                        axis=-1)
    lu = maximum(boxes1_t[..., :2], boxes2_t[..., :2])
    rd = minimum(boxes1_t[..., 2:], boxes2_t[..., 2:])

    intersection = maximum(0.0, rd - lu)
    inter_square = intersection[..., 0] * intersection[..., 1]

    square1 = boxes1[..., 2] * boxes1[..., 3]
    square2 = boxes2[..., 2] * boxes2[..., 3]

    #print("boxes:")
    # print(boxes1)
    # print(boxes2)

    # print("boxes T:")
    # print(boxes1_t)
    # print(boxes2_t)

    # print("first area")
    # print(square1)

    union_square = maximum(square1 + square2 - inter_square, 1e-10)
    return clip_by_value(inter_square / union_square, 0.0, 1.0)

def Difference(x, y):
    return tf.reduce_sum(tf.square(y - x))

def YoloLoss(HEIGHT, WIDTH, SPLIT_SIZE_1, SPLIT_SIZE_2):
    def YoloLoss_0(y_true, y_pred):
        target = y_true[..., 0]
        object_cell_numbers = tf.where(target[:]==1)

        #--------------------------------OBject Loss-----------------------------------------
        y_pred_extract = tf.gather_nd(y_pred, object_cell_numbers)
        y_target_extract = tf.gather_nd(y_true, object_cell_numbers)

        rescaler = object_cell_numbers
        upscaler_1 = tf.concat([rescaler[:, 1:], tf.zeros([len(rescaler), 2], dtype=tf.int64)], axis=-1)

        convert_coords_to_original_scale_tensor = tf.repeat([[WIDTH/SPLIT_SIZE_1, HEIGHT/SPLIT_SIZE_2, 1.0, 1.0]], repeats=[len(rescaler)], axis=0)

        target_upscaler_2 = tf.repeat([[1.0, 1.0, WIDTH, HEIGHT]], repeats=[len(rescaler)], axis=0) * tf.cast(y_target_extract[..., 1:5], dtype=tf.float32)
        pred_1box_upscaler_2 = tf.repeat([[1.0, 1.0, WIDTH, HEIGHT]], repeats=[len(rescaler)], axis=0) * tf.cast(y_pred_extract[..., 1:5], dtype=tf.float32)
        pred_2box_upscaler_2 = tf.repeat([[1.0, 1.0, WIDTH, HEIGHT]], repeats=[len(rescaler)], axis=0) * tf.cast(y_pred_extract[..., 6:10], dtype=tf.float32)

        original_target = (tf.cast(upscaler_1, dtype=tf.float32) + target_upscaler_2) * convert_coords_to_original_scale_tensor
        original_pred_1box = (tf.cast(upscaler_1, dtype=tf.float32) + pred_1box_upscaler_2) * convert_coords_to_original_scale_tensor
        original_pred_2box = (tf.cast(upscaler_1, dtype=tf.float32) + pred_2box_upscaler_2) * convert_coords_to_original_scale_tensor

        #print("real", original_target)
        #print("p_box1", original_pred_1box)
        #print("p_box2", original_pred_2box)

        #print("IoU with box1", ComputeIoU(original_target, original_pred_1box))

        mask = tf.cast(tf.math.greater(ComputeIoU(original_target, original_pred_2box), 
                ComputeIoU(original_target, original_pred_1box)), dtype=tf.int32)

        y_pred_only_confidences = tf.transpose(tf.concat([tf.expand_dims(y_pred_extract[..., 0], axis=0),
                tf.expand_dims(y_pred_extract[..., 5], axis=0)], axis=0))
      
        obj_pred = tf.gather_nd(y_pred_only_confidences, tf.stack([tf.range(len(rescaler)), mask], axis=-1))
          
        object_loss = Difference(tf.cast(obj_pred, dtype=tf.float32), tf.cast(tf.ones([len(rescaler)]), dtype=tf.float32))

        #--------------------------------For No object---------------------------------------
        y_pred_extract = tf.gather_nd(y_pred[..., 0:2*5], tf.where(target[:]==0))
        y_target_extract = tf.zeros(len(y_pred_extract))

        no_object_loss_1 = Difference(tf.cast(y_pred_extract[..., 0], dtype=tf.float32), tf.cast(y_target_extract, dtype=tf.float32))

        no_object_loss_2 = Difference(tf.cast(y_pred_extract[..., 5], dtype=tf.float32), tf.cast(y_target_extract, dtype=tf.float32))

        no_object_loss = no_object_loss_1 + no_object_loss_2

        #-----------------------------For OBject class loss---------------------------------
        y_pred_extract = tf.gather_nd(y_pred[..., 2*5:], tf.where(target[:]==1))
        class_extract = tf.gather_nd(y_true[..., 5:], tf.where(target[:]==1))

        class_loss = Difference(tf.cast(y_pred_extract, dtype=tf.float32), tf.cast(class_extract, dtype=tf.float32))

        #-------------------------For object bounding box loss-------------------------------
        y_pred_extract = tf.gather_nd(y_pred[..., 0:2*5], tf.where(target[:]==1))
        centre_joined = tf.stack([y_pred_extract[..., 1:3], y_pred_extract[..., 6:8]], axis=1)
        centre_pred = tf.gather_nd(centre_joined, tf.stack([tf.range(len(rescaler)), mask], axis=-1))
        centre_target = tf.gather_nd(y_true[..., 1:3], tf.where(target[:]==1))

        centre_loss = Difference(centre_pred, centre_target)

        size_joined=tf.stack([y_pred_extract[..., 3:5], y_pred_extract[..., 8:10]], axis=1)
        size_pred = tf.gather_nd(size_joined, tf.stack([tf.range(len(rescaler)), mask], axis=-1))
        size_target = tf.gather_nd(y_true[..., 3:5], tf.where(target[:]==1))

        size_loss = Difference(tf.math.sqrt(tf.math.abs(size_pred)), tf.math.sqrt(tf.math.abs(size_target)))
        
        box_loss = centre_loss + size_loss

        lambda_coord = 5.0
        lambda_no_obj = 0.5

        loss = object_loss + (lambda_no_obj * no_object_loss) + tf.cast(lambda_coord * box_loss, dtype=tf.float32) + tf.cast(class_loss, dtype=tf.float32) 
        return loss
    return YoloLoss_0


#testing methods
def GenerateOutput(bounding_boxes1, bounding_boxes2):
    SPLIT_SIZE_1 = 4
    SPLIT_SIZE_2 = 4
    CLASS_QUANTITY = 1
    output_label = np.zeros((SPLIT_SIZE_1, SPLIT_SIZE_2, CLASS_QUANTITY+10))
    for b in range(len(bounding_boxes1)):
        b1_grid_x = bounding_boxes1[..., b, 0] * SPLIT_SIZE_1
        b1_grid_y = bounding_boxes1[..., b, 1] * SPLIT_SIZE_2
        b1_i = int(b1_grid_x)
        b1_j = int(b1_grid_y)

        b2_grid_x = bounding_boxes2[..., b, 0] * SPLIT_SIZE_1
        b2_grid_y = bounding_boxes2[..., b, 1] * SPLIT_SIZE_2
        b2_i = int(b2_grid_x)
        b2_j = int(b2_grid_y)

        output_label[b1_i, b1_j, 0:5]=[0.95, b1_grid_x % 1, b1_grid_y % 1, bounding_boxes1[..., b, 2], bounding_boxes1[..., b, 3]]
        if b1_i == b2_i and b1_j == b2_j:
            output_label[b2_i, b2_j, 5:10]=[0.95, b2_grid_x % 1, b2_grid_y % 1, bounding_boxes2[..., b, 2], bounding_boxes2[..., b, 3]]
        output_label[b1_i, b1_j, 9+int(bounding_boxes1[..., b, 4])] = 1.

    return tf.expand_dims(tf.convert_to_tensor(output_label, tf.float32), axis=0)

def GenerateLabel(bounding_boxes):
    SPLIT_SIZE_1 = 4
    SPLIT_SIZE_2 = 4
    CLASS_QUANTITY = 1
    output_label = np.zeros((SPLIT_SIZE_1, SPLIT_SIZE_2, CLASS_QUANTITY+5))
    for b in range(len(bounding_boxes)):
        grid_x=bounding_boxes[..., b, 0] * SPLIT_SIZE_1
        grid_y=bounding_boxes[..., b, 1] * SPLIT_SIZE_2
        i = int(grid_x)
        j = int(grid_y)

        output_label[i, j, 0:5] = [1., grid_x % 1, grid_y % 1, bounding_boxes[..., b, 2], bounding_boxes[..., b, 3]]
        output_label[i, j, 4+int(bounding_boxes[..., b, 4])] = 1.

    return tf.expand_dims(tf.convert_to_tensor(output_label, tf.float32), axis=0)

def GenBox(box):
    width = 10
    height = 10
    xmin = box[0]
    ymin = box[1]
    xmax = box[2]
    ymax = box[3]
    bounding_box = [
        (xmin+xmax)/(2*width),(ymin+ymax)/(2*height),(xmax-xmin)/width,
        (ymax-ymin)/height, 1]
    return tf.convert_to_tensor(bounding_box)