

class ImageProcessing:

	@classmethod
	def DrawBoundingBoxOnImage(image,
                               ymin, xmin,
                               ymax, xmax,
                               color, font, thickness=4,
                               display_str_list=()):
		"""Adds a bounding box to an image."""
		draw = ImageDraw.Draw(image)
		width, height = image.size
		(left, right, top, bottom) = (xmin * width, xmax * width,
		                            ymin * height, ymax * height)
		draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
		         (left, top)], width=thickness, fill=color)

	@classmethod
	def DrawBoxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
		"""Overlay labeled boxes on an image with formatted scores and label names."""
		colors = list(ImageColor.colormap.values())
		font = ImageFont.load_default()

		for i in range(min(boxes.shape[0], max_boxes)):
			if scores[i] >= min_score:
				ymin, xmin, ymax, xmax = tuple(boxes[i])
				display_str = "{}: {}%".format(class_names[i].decode("ascii"),
				                            int(100 * scores[i]))
				color = colors[hash(class_names[i]) % len(colors)]
				image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
				DrawBoundingBoxOnImage(
					image_pil,
					ymin,
					xmin,
					ymax,
					xmax,
					color,
					font,
					display_str_list=[display_str])
				np.copyto(image, np.array(image_pil))
		return image