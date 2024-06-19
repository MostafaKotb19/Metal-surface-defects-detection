import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_anchor_boxes(n, scales, ratios, w, h):
    """
    Generate anchor boxes for object detection.
    
    Args:
    n (int): Number of anchors to generate per scale and ratio.
    scales (list): List of scales for the anchor boxes.
    ratios (list): List of aspect ratios for the anchor boxes.
    w (int): Width of the image.
    h (int): Height of the image.
    
    Returns:
    numpy.ndarray: An array of shape (n * len(scales) * len(ratios), 4)
                   containing the anchor boxes in (x1, y1, x2, y2) format.
    """
    
    center_x = np.linspace(0, w, n + 1)[:-1] + w / (2 * n)
    center_y = np.linspace(0, h, n + 1)[:-1] + h / (2 * n)
    cx, cy = np.meshgrid(center_x, center_y)
    
    centers = np.vstack([cx.ravel(), cy.ravel()]).transpose()
    
    anchor_boxes = []
    
    for (c_x, c_y) in centers:
        for scale in scales:
            for ratio in ratios:
                half_w = scale * np.sqrt(ratio) / 2
                half_h = scale / np.sqrt(ratio) / 2
                
                x1 = max(c_x - half_w, 0)
                y1 = max(c_y - half_h, 0)
                x2 = min(c_x + half_w, w)
                y2 = min(c_y + half_h, h)
                
                anchor_boxes.append([x1, y1, x2, y2]) 
    anchor_boxes = np.array(anchor_boxes)
    return anchor_boxes



def IoU(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
    box1 (list or numpy.ndarray): Coordinates of the first bounding box in format [x1, y1, x2, y2].
    box2 (list or numpy.ndarray): Coordinates of the second bounding box in format [x1, y1, x2, y2].

    Returns:
    float: Intersection over Union (IoU) score.
    """
    # Coordinates of intersection rectangle
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # Area of intersection rectangle
    intersection_area = max(0, x2_inter - x1_inter + 1) * max(0, y2_inter - y1_inter + 1)

    # Area of both bounding boxes
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Union area
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou 



def assign_anchor_boxes(y, anchor_boxes, lower_threshold=0.3, upper_threshold=0.5):
    """
    Assign each anchor box to a ground truth box.
    
    Args:
    y (numpy.ndarray): Ground truth bounding boxes of shape (n, 4).
    anchor_boxes (numpy.ndarray): Anchor boxes of shape (m, 4).
    lower_threshold (float): Lower threshold for IoU.
    upper_threshold (float): Upper threshold for IoU.

    Returns:
    numpy.ndarray: An array of shape (n, 1) containing the index of the assigned ground truth box for each anchor box.
    """
    
    m = anchor_boxes.shape[0]
    n = y.shape[0]
    
    iou_matrix = np.zeros((m, n))
    
    for i in range(m):
        for j in range(n):
            iou_matrix[i, j] = IoU(anchor_boxes[i], y[j])
    assigned_gt_idx = np.full(m, -1)
    iou_list = []
    for i in range(m):
        gt_idx = np.argmax(iou_matrix[i])
        iou = iou_matrix[i, gt_idx]
        iou_list.append(iou)
        if iou >= upper_threshold:
            assigned_gt_idx[i] = gt_idx
        elif iou < lower_threshold:
            assigned_gt_idx[i] = -1 # background
        else:
            assigned_gt_idx[i] = -2 # ignore
    # print(max(iou_list))
    return assigned_gt_idx



def calculate_offset(anchor_boxes, y, assigned_boxes):
    """
    Calculate the offset values for anchor boxes.

    Args:
    anchor_boxes (numpy.ndarray): Anchor boxes of shape (n, 4).
    y (numpy.ndarray): Ground truth bounding boxes of shape (n, 4).

    Returns:
    numpy.ndarray: Offset values for xyxy of shape (n, 4).
    """
    
    offsets = np.zeros_like(anchor_boxes)

    for i, box in enumerate(anchor_boxes):
        if assigned_boxes[i] >= 0:
            gt_box = y[assigned_boxes[i]]
            offsets[i] = [(gt_box[0] - box[0]) / (box[2] - box[0]),
                          (gt_box[1] - box[1]) / (box[3] - box[1]),
                          (gt_box[2] - box[2]) / (box[2] - box[0]),
                          (gt_box[3] - box[3]) / (box[3] - box[1])]
    return offsets



def assign_class_label(anchor_boxes, assigned_boxes, true_labels):
    """
    Assign class labels to anchor boxes.

    Args:
    anchor_boxes (numpy.ndarray): Anchor boxes of shape (n, 4).
    assigned_boxes (numpy.ndarray): Index of the assigned ground truth box for each anchor box.
    true_labels (numpy.ndarray): true labels for each ground truth box.

    Returns:
    numpy.ndarray: Class labels for each anchor box.
    """

    class_labels = np.full(anchor_boxes.shape[0], -1)
    for i, idx in enumerate(assigned_boxes):
        if idx >= 0:
            class_labels[i] = true_labels[idx]
        elif idx == -1:
            class_labels[i] = -1
        else:
            class_labels[i] = -2
    return class_labels



def format_gt(ground_truth, true_labels,  scales, ratios, n, w, h):
    """
    Formats the ground truth data by generating anchor boxes, assigning boxes to ground truth objects,
    and calculating offsets.

    Parameters:
    - ground_truth (list): List of ground truth objects.
    - scales (list): List of scales for anchor boxes.
    - ratios (list): List of ratios for anchor boxes.
    - w (int): Width of the image.
    - h (int): Height of the image.

    Returns:
    - offsets (list): List of offsets calculated for each anchor box.

    """

    anchor_boxes = generate_anchor_boxes(n, scales, ratios, w, h)
    assigned_boxes = assign_anchor_boxes(ground_truth, anchor_boxes)
    offsets = calculate_offset(anchor_boxes, ground_truth, assigned_boxes)
    class_labels = assign_class_label(anchor_boxes, assigned_boxes, true_labels)

    return offsets, class_labels
