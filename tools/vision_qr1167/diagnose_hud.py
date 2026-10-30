"""Offline font-ink diagnostic, deliberately NOT an observation admission gate."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image


def candidates(calibration, rgb):
    frames=[]
    templates=[]
    for digit in range(10):
        frame=np.asarray(Image.open(calibration/f'camera-{10000+digit*110}.png').convert('RGB'))
        frames.append(frame)
        crop=frame[798:845,825:849]
        white=crop.min(2)>242
        dilated=np.zeros_like(white)
        for dy in range(-2,3):
            for dx in range(-2,3):
                dilated|=np.roll(np.roll(white,dy,0),dx,1)
        black=(crop.max(2)<90)&dilated&~white
        templates.append((white,black))
    result=[]
    for slot,origin in enumerate((686,710,745,768,801,825)):
        local_templates=list(templates)
        for d,frame in enumerate(frames):
            time_digits=[0,0,1,0,d,d] # actual calibration: 10.00 + digit*110ms
            value=10000+d*110
            time_digits=[0,0,value//10000,(value//1000)%10,(value//100)%10,(value//10)%10]
            digit=time_digits[slot]
            crop=frame[798:845,origin:origin+24]
            white=crop.min(2)>242
            dilated=np.zeros_like(white)
            for dy in range(-2,3):
                for dx in range(-2,3):dilated|=np.roll(np.roll(white,dy,0),dx,1)
            black=(crop.max(2)<90)&dilated&~white
            local_templates[digit]=(white,black)
        scores=[]
        for white,black in local_templates:
            scores.append(min(float(.5*np.mean((rgb[798:845,origin+shift:origin+shift+24].min(2)<=242)[white])+
                .5*np.mean((rgb[798:845,origin+shift:origin+shift+24].max(2)>=90)[black])) for shift in (-1,0,1)))
        order=np.argsort(scores)
        result.append(dict(candidate=int(order[0]),error=scores[order[0]],
                           margin=scores[order[1]]-scores[order[0]]))
    return result


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('calibration',type=Path)
    p.add_argument('images',nargs='+',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report=dict(status='DIAGNOSTIC_ONLY_NOT_ADMITTED',frames=[])
    for path in a.images:
        rgb=np.asarray(Image.open(path).convert('RGB'))
        if rgb.shape!=(864,1536,3):raise ValueError('image dimensions')
        report['frames'].append(dict(image=str(path),slots=candidates(a.calibration,rgb)))
    with a.output.open('x') as file:json.dump(report,file,indent=2)


if __name__=='__main__':main()
