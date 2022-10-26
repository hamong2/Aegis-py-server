import cv2    
import numpy as np
CLASSES = [ # persen의 obj_index는 0
     'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',  'backpack',
    'umbrella',   'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle',  'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed',  'dining table', 
     'toilet',  'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier','toothbrush',

    'middle_finger', 'cigarret', 'food' 
]

ACTIONS = ['adjust', 'assemble', 'block', 'blow', 'board', 'break', 'brush_with', 'buy', 'carry', 'catch', 
           'chase', 'check', 'clean', 'control', 'cook', 'cut', 'cut_with', 'direct', 'drag', 'dribble', 
            'drink_with', 'drive', 'dry', 'eat', 'eat_at', 'exit', 'feed', 'fill', 'flip', 'flush', 'fly',
            'greet', 'grind', 'groom', 'herd', 'hit', 'hold', 'hop_on', 'hose', 'hug', 'hunt', 'inspect', 
            'install', 'jump', 'kick', 'kiss', 'lasso', 'launch', 'lick', 'lie_on', 'lift', 'light', 'load', 
            'lose', 'make', 'milk', 'move', 'no_interaction', 'open', 'operate', 'pack', 'paint', 'park', 'pay',
            'peel', 'pet', 'pick', 'pick_up', 'point', 'pour', 'pull', 'push', 'race', 'read', 'release', 
            'repair', 'ride', 'row', 'run', 'sail', 'scratch', 'serve', 'set', 'shear', 'sign', 'sip', 
            'sit_at', 'sit_on', 'slide', 'smell', 'spin', 'squeeze', 'stab', 'stand_on', 'stand_under', 
            'stick', 'stir', 'stop_at', 'straddle', 'swing', 'tag', 'talk_on', 'teach', 'text_on', 'throw', 
            'tie', 'toast', 'train', 'turn', 'type_on', 'walk', 'wash', 'watch', 'wave', 'wear', 'wield', 'zip', 

            'fuck_you', 'smoke', 'stab', 'strangle', 'violence'
            ]

  
# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

need_mosaic = ['fuck_you', 'smoke', 'stab', 'strangle', 'violence']
harmful_obj = ['cigarret', 'middle_finger']

ksize = 50
def plot_results(frame, prob, boxes, verbs):
    frame = np.array(frame)
    frame = np.uint8(frame)
    for p, (x1, y1, x2, y2), c in zip(prob, boxes, COLORS * 100):
        x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
        x_2 = max(x1, x2)
        x_1 = min(x1, x2)
        y_2 = max(y1, y2)
        y_1 = min(y1, y2)
        if x_1 < 0:
          x_1 = 0
        if y_1 < 0:
          y_1=0
        # print(x_2, x_1, y_2, y_1)

        roi = frame[y_1:y_2, x_1:x_2]  
        roi = cv2.blur(roi, (ksize, ksize))
        frame[y_1:y_2, x_1:x_2] = roi   # 원본 이미지에 적용
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,0), 3)
        text = f'{CLASSES[p]}'
        cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)        

    for i,verb in enumerate(verbs):
        xmin, ymin, xmax, ymax = boxes[i]
        xmin2, ymin2, xmax2, ymax2 = boxes[i+1]
        x1, x2 = (xmin+xmax)/2, (xmin2+xmax2)/2
        y1, y2 = (ymin+ymax)/2, (ymin2+ymax2)/2
        x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
        cv2.circle(frame, (x1, y1), 5, (0,255,0), -1)
        cv2.circle(frame, (x2, y2), 5, (0,255,0), -1)
        cv2.line(frame, (x1, y1), (x2, y2), (0,255,0))
        text = f'{ACTIONS[verb]}'
        cv2.putText(frame, text, (int((x1+x2)/2), int((y1+y2)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)         

    return frame


need_mosaic = ['fuck_you', 'smoke', 'stab', 'strangle', 'violence']

def timelog(verbs):
  violence = False
  for i,verb in enumerate(verbs):
    if ACTIONS[verb] in need_mosaic:
      violence = True
      break
  return violence

# to get harmful action
def outputaction(verbs):
  for i,verb in enumerate(verbs):
    if ACTIONS[verb] in need_mosaic:
      output = ACTIONS[verb]
      break
  return output

