import argparse
import random
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
from models import build_model

from aiohttp import web
import socketio
import time
from args import get_args_parser

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

             'violence', 'strangle', 'smoke', 'cook', 'stabe'
            ]

sio = socketio.AsyncServer(cors_allowed_origins="*",
                           logger=False,
                           max_http_buffer_size=10**18)
app = web.Application()
sio.attach(app)
parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
args = parser.parse_args()
device = torch.device(args.device)
seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

model, postprocessors = build_model(args)
model.to(device)

checkpoint = torch.load(args.pretrained, map_location="cuda")
model.load_state_dict(checkpoint['model'], strict=False)

orig_target_sizes =  torch.as_tensor([[300, 400]])
transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
cnt = 0
start = 0
@sio.event
def connect(sid, environ):
	print("connect ", sid)

@sio.event
def disconnect(sid):
	print("disconnect ", sid)
 
@sio.on('filtering')
async def filter_image(sid, data):
    global cnt, start
    if start == 0:
        start = time.time()
    count = data['count']
    if count < 5:
        await sio.emit('filter', {# 'img': data['origin'], 
            'bbox': [], 'verb': -1, 'count': count, 'time': data['time']})
    else:
        t = time.time()
        image = data['rgb']
        img = Image.frombytes(mode="RGB", size=(300, 400), data=image)
        img = transform(img).unsqueeze(0).to(device)
        outputs = model(img) 
        preds = postprocessors['hoi'](outputs, orig_target_sizes)
        bbox = preds[0]['boxes']
        score = preds[0]['verb_scores'] # 100, 117
        actions = score.max(-1)
        idx = np.argmax(actions.values)
        if actions.values[idx] < 0.2:
            await sio.emit('filter', {'bbox': [0,0,0,0], 'verb': 0, 'count': count, 'time': data['time']})
        else:
            box = []
            box.append(bbox[idx].tolist())
            box.append(bbox[idx+100].tolist())
            act = actions.indices[idx]
            verb = []
            verb.append(ACTIONS[act])
            ind = ACTIONS.index(verb[0])
            print(verb)
            await sio.emit('filter', {'bbox': box, 'verb': ind, 'count': count, 'time': data['time']})


web.run_app(app, host="127.0.0.1", port=7080)
# fix the seed for reproducibility

