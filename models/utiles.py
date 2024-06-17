from models.loss import ComputeIoU
import tensorflow as tf
from settings import *

def CustomMetrics(WIDTH, HEIGHT, SPLIT_SIZE_1, SPLIT_SIZE_2, CLASS_QUANTITY):
	metrics_for_each_class = []
	def PrecisionAndRecallForOneClass(i):
		IoU_threshold = 0.5
		C_threshold = 0.25
		def Precision(label, prediction):
			TP = tf.constant(0); FP = tf.constant(0); FN = tf.constant(0)

			real_i = label[..., 5 + i]
			all_object_cells_with_class_i = tf.where(real_i[:]==1) #rescaler for x y based on i j of the cell

			real_extract = tf.gather_nd(label, all_object_cells_with_class_i)
			prediction_extract = tf.gather_nd(prediction, all_object_cells_with_class_i)
			positions = all_object_cells_with_class_i[:, 1:]

			N = len(all_object_cells_with_class_i)

			first_conf_c_thresh = tf.math.greater(prediction_extract[..., 0], C_threshold)
			second_conf_c_thresh = tf.math.greater(prediction_extract[..., 5], C_threshold)
			both_conf_c_thresh = tf.math.logical_or(first_conf_c_thresh, second_conf_c_thresh)

			indexes_conf_normal = tf.where(both_conf_c_thresh[:])

			real_extract = tf.gather_nd(real_extract, indexes_conf_normal)
			prediction_extract = tf.gather_nd(prediction_extract, indexes_conf_normal)
			positions = tf.gather_nd(positions, indexes_conf_normal)

			mask = tf.cast(tf.math.greater(prediction_extract[..., 5], prediction_extract[..., 0]), dtype=tf.int32)

			K = len(indexes_conf_normal)
			base = tf.concat([positions, tf.zeros([K, 2], dtype=tf.int64)], axis=-1)
			convert_coords_to_original_scale_tensor = [[WIDTH/SPLIT_SIZE_1, HEIGHT/SPLIT_SIZE_2, 1.0, 1.0]]
			
			real_upscaler = tf.repeat([[1.0, 1.0, WIDTH, HEIGHT]], repeats=[K], axis=0) * tf.cast(real_extract[..., 1:5], dtype=tf.float32)
			pred_1box_upscaler = tf.repeat([[1.0, 1.0, WIDTH, HEIGHT]], repeats=[K], axis=0) * tf.cast(prediction_extract[..., 1:5], dtype=tf.float32)
			pred_2box_upscaler = tf.repeat([[1.0, 1.0, WIDTH, HEIGHT]], repeats=[K], axis=0) * tf.cast(prediction_extract[..., 6:10], dtype=tf.float32)
			
			original_real_box = (tf.cast(base, dtype=tf.float32) + real_upscaler) * tf.repeat(convert_coords_to_original_scale_tensor, repeats=[K], axis=0)
			original_pred_1box = (tf.cast(base, dtype=tf.float32) + pred_1box_upscaler) * tf.repeat(convert_coords_to_original_scale_tensor, repeats=[K], axis=0)
			original_pred_2box = (tf.cast(base, dtype=tf.float32) + pred_2box_upscaler) * tf.repeat(convert_coords_to_original_scale_tensor, repeats=[K], axis=0)

			IoUreal_box1 = ComputeIoU(original_real_box, original_pred_1box)
			IoUreal_box2 = ComputeIoU(original_real_box, original_pred_2box)

			prediction_IoUs = tf.transpose(tf.concat([tf.expand_dims(IoUreal_box1, axis=0),
	                tf.expand_dims(IoUreal_box2, axis=0)], axis=0))

			indexes_of_best = tf.stack([tf.range(K), mask], axis=-1)
			IoUs_of_best_boxes = tf.gather_nd(prediction_IoUs, indexes_of_best)
			full_confidences = IoUs_of_best_boxes

			confidence_indexes = tf.where(full_confidences[:] > IoU_threshold)
			M = len(confidence_indexes)
			prediction_extract = tf.gather_nd(prediction_extract, confidence_indexes)
			predicted_classes = tf.cast(tf.math.argmax(prediction_extract[..., 10:], axis=1), tf.uint32) #classes of predictions which correspond to real i-class labels
			res = tf.math.equal(tf.constant(i, dtype=tf.uint32), predicted_classes)
			TP = tf.math.reduce_sum(tf.cast(res, tf.int32))

			all_no_object_cells_with_class_i = tf.where(real_i[:]==0)
			prediction_extract = tf.gather_nd(prediction, all_no_object_cells_with_class_i)

			first_conf_c_thresh = tf.math.greater(prediction_extract[..., 0], C_threshold)
			second_conf_c_thresh = tf.math.greater(prediction_extract[..., 5], C_threshold)
			both_conf_c_thresh = tf.math.logical_or(first_conf_c_thresh, second_conf_c_thresh)

			indexes_conf_normal = tf.where(both_conf_c_thresh[:])
			prediction_extract = tf.gather_nd(prediction_extract, indexes_conf_normal)

			prediction_class = tf.argmax(prediction_extract[..., 10:], axis=1)
			mask = tf.equal(prediction_class, tf.constant(i, dtype=tf.int64))
			FP = tf.math.reduce_sum(tf.cast(mask, tf.int32))
			
			TP_FP = tf.math.add(TP, FP)
			Precision = TP / TP_FP if TP_FP > 0 else tf.constant(0.0, dtype=tf.float64)
			return Precision

		def Recall(label, prediction):
			TP = tf.constant(0); FP = tf.constant(0); FN = tf.constant(0)

			real_i = label[..., 5 + i]
			all_object_cells_with_class_i = tf.where(real_i[:]==1) #rescaler for x y based on i j of the cell

			real_extract = tf.gather_nd(label, all_object_cells_with_class_i)
			prediction_extract = tf.gather_nd(prediction, all_object_cells_with_class_i)
			positions = all_object_cells_with_class_i[:, 1:]

			N = len(all_object_cells_with_class_i)

			first_conf_c_thresh = tf.math.greater(prediction_extract[..., 0], C_threshold)
			second_conf_c_thresh = tf.math.greater(prediction_extract[..., 5], C_threshold)
			both_conf_c_thresh = tf.math.logical_or(first_conf_c_thresh, second_conf_c_thresh)

			indexes_conf_normal = tf.where(both_conf_c_thresh[:])

			real_extract = tf.gather_nd(real_extract, indexes_conf_normal)
			prediction_extract = tf.gather_nd(prediction_extract, indexes_conf_normal)
			positions = tf.gather_nd(positions, indexes_conf_normal)

			mask = tf.cast(tf.math.greater(prediction_extract[..., 5], prediction_extract[..., 0]), dtype=tf.int32)

			K = len(indexes_conf_normal)
			base = tf.concat([positions, tf.zeros([K, 2], dtype=tf.int64)], axis=-1)
			convert_coords_to_original_scale_tensor = [[WIDTH/SPLIT_SIZE_1, HEIGHT/SPLIT_SIZE_2, 1.0, 1.0]]
			
			real_upscaler = tf.repeat([[1.0, 1.0, WIDTH, HEIGHT]], repeats=[K], axis=0) * tf.cast(real_extract[..., 1:5], dtype=tf.float32)
			pred_1box_upscaler = tf.repeat([[1.0, 1.0, WIDTH, HEIGHT]], repeats=[K], axis=0) * tf.cast(prediction_extract[..., 1:5], dtype=tf.float32)
			pred_2box_upscaler = tf.repeat([[1.0, 1.0, WIDTH, HEIGHT]], repeats=[K], axis=0) * tf.cast(prediction_extract[..., 6:10], dtype=tf.float32)
			
			original_real_box = (tf.cast(base, dtype=tf.float32) + real_upscaler) * tf.repeat(convert_coords_to_original_scale_tensor, repeats=[K], axis=0)
			original_pred_1box = (tf.cast(base, dtype=tf.float32) + pred_1box_upscaler) * tf.repeat(convert_coords_to_original_scale_tensor, repeats=[K], axis=0)
			original_pred_2box = (tf.cast(base, dtype=tf.float32) + pred_2box_upscaler) * tf.repeat(convert_coords_to_original_scale_tensor, repeats=[K], axis=0)

			IoUreal_box1 = ComputeIoU(original_real_box, original_pred_1box)
			IoUreal_box2 = ComputeIoU(original_real_box, original_pred_2box)

			prediction_IoUs = tf.transpose(tf.concat([tf.expand_dims(IoUreal_box1, axis=0),
	                tf.expand_dims(IoUreal_box2, axis=0)], axis=0))

			indexes_of_best = tf.stack([tf.range(K), mask], axis=-1)
			IoUs_of_best_boxes = tf.gather_nd(prediction_IoUs, indexes_of_best)
			full_confidences = IoUs_of_best_boxes

			confidence_indexes = tf.where(full_confidences[:] > IoU_threshold)
			M = len(confidence_indexes)
			prediction_extract = tf.gather_nd(prediction_extract, confidence_indexes)
			predicted_classes = tf.cast(tf.math.argmax(prediction_extract[..., 10:], axis=1), tf.uint32) #classes of predictions which correspond to real i-class labels
			res = tf.math.equal(tf.constant(i, dtype=tf.uint32), predicted_classes)
			TP = tf.math.reduce_sum(tf.cast(res, tf.int32))
			FN = tf.math.subtract(N, TP)                #FN = FN + M - TP: IoU - ok, class - not ok
			               								#FN = FN + N - M: IoU - not ok
			
			TP_FN = tf.math.add(TP, FN)
			Recall = TP / TP_FN if TP_FN > 0 else tf.constant(0.0, dtype=tf.float64)
			return Recall

		Precision.__name__ = 'Precision_' + str(i)
		Recall.__name__ = 'Recall_' + str(i)
		return [Precision, Recall]

	for i in range(0, CLASS_QUANTITY):
		metrics_for_each_class.extend(PrecisionAndRecallForOneClass(i))

	#return tf.cast(metrics_for_each_class, dtype=tf.float32)
	#return PrecisionAndRecall
	return metrics_for_each_class

def MakeExtractor(model_type):
	WIDTH = MODEL_PARAMS[MT(model_type)]["WIDTH"]
	HEIGHT = MODEL_PARAMS[MT(model_type)]["HEIGHT"]
	SPLIT_SIZE_1 = MODEL_PARAMS[MT(model_type)]["SPLIT_SIZE_1"]
	SPLIT_SIZE_2 = MODEL_PARAMS[MT(model_type)]["SPLIT_SIZE_2"]
	def GetBoxFromPrediction(pred):
		pred_conf1 = pred[..., 0]
		pred_conf2 = pred[..., 5]
		condition = tf.math.logical_or(pred_conf1[:] >= 0.25, pred_conf2[:] >= 0.25)
		indexes = tf.where(condition)

		prediction_extract = tf.gather_nd(pred, indexes) #tf.reshape(pred, [1, pred.shape[pred.ndim-1]])
		#print(prediction_extract)

		prediction_conf1 = prediction_extract[..., 0]
		prediction_conf2 = prediction_extract[..., 5]

		max_conf = tf.cast(tf.math.greater(prediction_conf2, prediction_conf1), tf.int32)
		all_boxes = tf.stack([prediction_extract[..., 1:5], prediction_extract[..., 6:10]], axis=1)
		predicted_class = tf.cast(tf.argmax(prediction_extract[..., 10:], axis=1), tf.float32)
		#print("all boxes: ", all_boxes)
		best_boxes = tf.gather_nd(all_boxes, tf.transpose(tf.stack([tf.range(len(all_boxes)), max_conf], axis=0)))
		#print("best boxes: ", best_boxes)

		N = len(indexes)
		base = tf.concat([indexes[:, 1:], tf.zeros([N, 2], dtype=tf.int64)], axis=-1)
		convert_coords_to_original_scale_tensor = [[WIDTH/SPLIT_SIZE_1, HEIGHT/SPLIT_SIZE_2, 1.0, 1.0]]
				
		pred_box_upscaler = tf.repeat([[1.0, 1.0, WIDTH, HEIGHT]], repeats=[N], axis=0) * best_boxes

		original_boxes = (tf.cast(base, dtype=tf.float32) + pred_box_upscaler) * tf.repeat(convert_coords_to_original_scale_tensor, repeats=[N], axis=0)

		#convert to angles!!!
		original_boxes_in_angle_format = tf.stack([original_boxes[..., 0] - original_boxes[..., 2] / 2.0,
                         original_boxes[..., 1] - original_boxes[..., 3] / 2.0,
                         original_boxes[..., 0] + original_boxes[..., 2] / 2.0,
                         original_boxes[..., 1] + original_boxes[..., 3] / 2.0,
                         predicted_class],
                        axis=-1)
		return tf.clip_by_value(tf.cast(original_boxes_in_angle_format, tf.int32), clip_value_min = 0, clip_value_max = HEIGHT - 1)

	return GetBoxFromPrediction

