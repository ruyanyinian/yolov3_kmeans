# ============================================================
#   Copyright (c) 2022, EZXR Inc. All rights reserved
#   File        : kmeans.py
#   Author      : qinyu@ezxr.com
#   Created date: 2022/7/7 11:02
#   Description : 
# ============================================================
from voc import VOCDetection
import numpy as np
import random
import argparse
import os
import sys


def parse_args():
  parser = argparse.ArgumentParser(description='kmeans for anchor box')

  parser.add_argument('-root', '--root', default=r'F:\DL_Data\image_detection\VOC\train',
                      help='dataset root')
  parser.add_argument('-d', '--dataset', default='voc',
                      help='coco, voc.')
  parser.add_argument('-na', '--num_anchorbox', default=9, type=int,
                      help='number of anchor box.')
  parser.add_argument('-size', '--input_size', default=416, type=int,
                      help='input size.')
  parser.add_argument('--scale', action='store_true', default=False,
                      help='divide the sizes of anchor boxes by 32 .')
  return parser.parse_args()


args = parse_args()


class Box():
  def __init__(self, x, y, w, h):
    self.x = x
    self.y = y
    self.w = w
    self.h = h


def iou(box1, box2):
  x1, y1, w1, h1 = box1.x, box1.y, box1.w, box1.h
  x2, y2, w2, h2 = box2.x, box2.y, box2.w, box2.h

  S_1 = w1 * h1
  S_2 = w2 * h2

  xmin_1, ymin_1 = x1 - w1 / 2, y1 - h1 / 2
  xmax_1, ymax_1 = x1 + w1 / 2, y1 + h1 / 2
  xmin_2, ymin_2 = x2 - w2 / 2, y2 - h2 / 2
  xmax_2, ymax_2 = x2 + w2 / 2, y2 + h2 / 2

  I_w = min(xmax_1, xmax_2) - max(xmin_1, xmin_2)
  I_h = min(ymax_1, ymax_2) - max(ymin_1, ymin_2)
  if I_w < 0 or I_h < 0:
    return 0
  I = I_w * I_h

  IoU = I / (S_1 + S_2 - I)

  return IoU


def init_centroids(boxes, n_anchors):
  """
      We use kmeans++ to initialize centroids.
  """
  centroids = []
  boxes_num = len(boxes)

  centroid_index = int(np.random.choice(boxes_num, 1)[0]) # 随机选取一个index
  centroids.append(boxes[centroid_index])
  print(centroids[0].w, centroids[0].h) # 在416*416尺寸上的w和h

  for centroid_index in range(0, n_anchors - 1):
    sum_distance = 0
    distance_thresh = 0
    distance_list = []
    cur_sum = 0

    for box in boxes:
      min_distance = 1
      for centroid_i, centroid in enumerate(centroids): # centroid是box对象
        distance = (1 - iou(box, centroid)) # 这里是每一个GT的bbox和一开始选取的GTbbox进行iOU的计算
        if distance < min_distance:
          min_distance = distance
      sum_distance += min_distance
      distance_list.append(min_distance)

    distance_thresh = sum_distance * np.random.random() # 这里选取一个0~1的随机值。

    for i in range(0, boxes_num):
      cur_sum += distance_list[i]
      if cur_sum > distance_thresh: # 这里进行累加, 如果距离大于distance threshold, 那么就添加对应的centroids
        centroids.append(boxes[i])
        print(boxes[i].w, boxes[i].h)
        break
  return centroids


def do_kmeans(n_anchors, boxes, centroids):
  loss = 0
  groups = []
  new_centroids = []
  # for box in centroids:
  #     print('box: ', box.x, box.y, box.w, box.h)
  # exit()
  for i in range(n_anchors):
    groups.append([])  # 生成9个空的list
    new_centroids.append(Box(0, 0, 0, 0))  # 生成9个空的box

  for box in boxes: # 遍历40000+的bbox
    min_distance = 1
    group_index = 0
    for centroid_index, centroid in enumerate(centroids):  # 遍历9个centroids的bbox
      distance = (1 - iou(box, centroid))
      if distance < min_distance:
        min_distance = distance
        group_index = centroid_index

    groups[group_index].append(box)
    loss += min_distance  # 这里的loss就是min_distance
    new_centroids[group_index].w += box.w
    new_centroids[group_index].h += box.h

  for i in range(n_anchors):
    new_centroids[i].w /= max(len(groups[i]), 1)
    new_centroids[i].h /= max(len(groups[i]), 1)

  return new_centroids, groups, loss  # / len(boxes)


def anchor_box_kmeans(total_gt_boxes, n_anchors, loss_convergence, iters, plus=True):
  """
  This function will use k-means to get appropriate anchor boxes for train dataset.
  Input:
    total_gt_boxes:
    n_anchor : int -> the number of anchor boxes.
    loss_convergence : float -> threshold of iterating convergence.
    iters: int -> the number of iterations for training kmeans.
  Output: anchor_boxes : list -> [[w1, h1], [w2, h2], ..., [wn, hn]].
  """
  boxes = total_gt_boxes # list of boxes, length = 40058, 这些坐标是映射到(416,416上面)
  centroids = []
  if plus:
    # list of boxes, length = 40058, n_anchors=9, 初始化9个centroids, 一个centroid是一个bbox对象
    centroids = init_centroids(boxes, n_anchors)
  else:
    total_indexs = range(len(boxes))
    sample_indexs = random.sample(total_indexs, n_anchors)
    for i in sample_indexs:
      centroids.append(boxes[i])

  # iterate k-means
  centroids, groups, old_loss = do_kmeans(n_anchors, boxes, centroids) # n_anchors=9, centroids其实是list of 9个boxes
  iterations = 1
  while (True):
    centroids, groups, loss = do_kmeans(n_anchors, boxes, centroids)
    iterations += 1
    print("Loss = %f" % loss)
    if abs(old_loss - loss) < loss_convergence or iterations > iters:
      break
    old_loss = loss

    for centroid in centroids:
      print(centroid.w, centroid.h)

  print("k-means result : ")
  for centroid in centroids:
    if args.scale:
      print("w, h: ", round(centroid.w / 32., 2), round(centroid.h / 32., 2),
            "area: ", round(centroid.w / 32., 2) * round(centroid.h / 32., 2))
    else:
      print("w, h: ", round(centroid.w, 2), round(centroid.h, 2),
            "area: ", round(centroid.w, 2) * round(centroid.h, 2))

  return centroids


if __name__ == "__main__":

  n_anchors = args.num_anchorbox
  img_size = args.input_size
  dataset = args.dataset

  loss_convergence = 1e-6
  iters_n = 1000

  dataset_voc = VOCDetection(data_dir=os.path.join(args.root, 'VOCdevkit'))

  boxes = []
  print("The dataset size: ", len(dataset))
  print("Loading the dataset ...")
  # VOC
  for i in range(len(dataset_voc)):
    if i % 5000 == 0:
      print('Loading voc data [%d / %d]' % (i+1, len(dataset_voc)))

    # For VOC
    img, _ = dataset_voc.pull_image(i)
    w, h = img.shape[1], img.shape[0]
    _, annotation = dataset_voc.pull_anno(i)  # [[262.0, 210.0, 323.0, 338.0, 8], (xmin, ymin, xmax, ymax, label)[164.0, 263.0, 252.0, 371.0, 8], [240.0, 193.0, 294.0, 298.0, 8]] 一张图像的bbox的信息

    # prepare bbox datas
    for box_and_label in annotation:
      box = box_and_label[:-1]  # 坐标框
      xmin, ymin, xmax, ymax = box

      # 将坐标映射到(416,416)的图像上
      bw = (xmax - xmin) / w * img_size  # img_size = 416
      bh = (ymax - ymin) / h * img_size

      # check bbox
      if bw < 1.0 or bh < 1.0:
        continue
      boxes.append(Box(0, 0, bw, bh))

  print("Number of all bboxes: ", len(boxes))
  print("Start k-means !")
  centroids = anchor_box_kmeans(boxes, n_anchors, loss_convergence, iters_n, plus=True)

