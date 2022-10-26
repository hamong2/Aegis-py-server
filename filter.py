import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import cv2

import json
from models import build_model
from args import get_args_parser
from plot import plot_results, timelog
from flask import Flask, request, Response, send_file 
from flask_cors import CORS, cross_origin
from requests_toolbelt import MultipartEncoder

parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
args = parser.parse_args()
if args.output_dir:
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

device = torch.device(args.device)

seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

model, postprocessors = build_model(args)
model.to(device)

checkpoint = torch.load('./checkpoint.pth', map_location='cuda')
model.load_state_dict(checkpoint['model'], strict=False)

transform = T.Compose([
        T.ToPILImage(),
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])      

app = Flask(__name__)
CORS(app)
@app.route('/upload', methods=["GET","POST"])
def upload_file():
    if request.method == "POST":
        print(request)
        f = request.files['file']
        print(f)
        f.save('./tmp.mp4')
        time_log = []
        cap = cv2.VideoCapture('./tmp.mp4')
        print(cap)
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        orig_target_sizes = torch.as_tensor([[int(frame_height), int(frame_width)]])
        out = cv2.VideoWriter('./vfilter.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 30.0, (int(w), int(h)))
        print(f)        
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                frames = transform(frame).unsqueeze(0).to(device)
                outputs = model(frames)
                preds =  postprocessors['hoi'](outputs, orig_target_sizes)
                labels = preds[0]['labels']
                box = preds[0]['boxes']
                score = preds[0]['verb_scores']
                actions = score.max(-1)
                idxs = []
                for i, action in enumerate(actions.values):
                    if action > 0.125:
                        if actions.indices[i] == 118:
                            idxs.append(i)
                            break
                boxes = []
                labelss = []
                verb = []
                for i in idxs:
                    boxes.append(box[i])
                    boxes.append(box[i+100])
                    labelss.append(labels[i])
                    labelss.append(labels[i+100])
                    verb.append(actions.indices[i])

                frame = plot_results(frame, labelss, boxes, verb)
                out.write(frame)
                if timelog(verb) == True:
                    t = cap.get(cv2.CAP_PROP_POS_MSEC)
                    time_log.append(int(t*1000))
                if cv2.waitKey(27) & 0xFF == 27:
                    break
            else:
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    return Response(json.dumps({'log':time_log}), mimetype="application/json", status=200)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=7070)
