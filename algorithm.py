def ensemble_net(image):
  """
  This function takes an image as input and outputs bounding boxes for detected classes along with their objectness score.

  Args:
    image: The input image.

  Returns:
    A list of bounding boxes for detected classes along with their objectness score.
  """

    # Set the IoU threshold for ensembling.
  threshold = 0.5

  # Get detections from Faster R-CNN and YOLOv5 models.
  faster_rcnn_detections = get_faster_rcnn_detections(image)
  yolov5_detections = get_yolov5_detections(image)

  # Initialize empty list to store ensembled detections.
  ensembled_detections = []

  # Loop through each Faster R-CNN detection.
  for faster_rcnn_detection in faster_rcnn_detections:
    faster_rcnn_bbox, faster_rcnn_class, faster_rcnn_score = faster_rcnn_detection

    # Find the corresponding YOLOv5 detection for the same class.
    corresponding_yolov5_detection = None
    for yolov5_detection in yolov5_detections:
      yolov5_bbox, yolov5_class, yolov5_score = yolov5_detection
      if yolov5_class == faster_rcnn_class:
        corresponding_yolov5_detection = yolov5_detection
        break

    # If there is a corresponding YOLOv5 detection.
    if corresponding_yolov5_detection:
      yolov5_bbox, yolov5_class, yolov5_score = corresponding_yolov5_detection

      # Check if the bounding boxes are close enough (IoU > threshold).
      if is_iou_greater_than_threshold(faster_rcnn_bbox, yolov5_bbox, threshold):
        # If the Faster R-CNN score is higher, use the Faster R-CNN detection.
        if faster_rcnn_score > yolov5_score:
          ensembled_detections.append([faster_rcnn_bbox, faster_rcnn_class, faster_rcnn_score])
        # Else, use the YOLOv5 detection.
        else:
          ensembled_detections.append([yolov5_bbox, yolov5_class, yolov5_score])
      # Else, if the bounding boxes are not close enough, use the detection with the higher score.
      else:
        if faster_rcnn_score > yolov5_score:
          ensembled_detections.append([faster_rcnn_bbox, faster_rcnn_class, faster_rcnn_score])
        else:
          ensembled_detections.append([yolov5_bbox, yolov5_class, yolov5_score])
    # Else, if there is no corresponding YOLOv5 detection, use the Faster R-CNN detection.
    else:
      ensembled_detections.append([faster_rcnn_bbox, faster_rcnn_class, faster_rcnn_score])

  return ensembled_detections

