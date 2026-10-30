"""Position-calibrated white-ink/black-outline clock, independent of telemetry."""
from pathlib import Path
import numpy as np
from PIL import Image


class InkClock:
    origins=(686,710,745,768,801,825)

    def __init__(self, calibration):
        frames=[np.asarray(Image.open(Path(calibration)/f'camera-{10000+d*110}.png').convert('RGB')) for d in range(10)]
        common=[self.pattern(frame[798:845,825:849]) for frame in frames]
        self.templates=[]
        for slot,origin in enumerate(self.origins):
            templates=list(common)
            for d,frame in enumerate(frames):
                value=10000+d*110
                digits=(0,0,value//10000,(value//1000)%10,(value//100)%10,(value//10)%10)
                templates[digits[slot]]=self.pattern(frame[798:845,origin:origin+24])
            self.templates.append(templates)

    @staticmethod
    def pattern(crop):
        white=crop.min(2)>242
        dilated=np.zeros_like(white)
        for dy in range(-2,3):
            for dx in range(-2,3):dilated|=np.roll(np.roll(white,dy,0),dx,1)
        black=(crop.max(2)<90)&dilated&~white
        if not white.any() or not black.any():raise ValueError('empty font calibration')
        return white,black

    def read(self,rgb):
        if rgb.shape!=(864,1536,3):raise ValueError('ink clock dimensions')
        digits=[];errors=[]
        for origin,templates in zip(self.origins,self.templates):
            scores=[]
            for white,black in templates:
                scores.append(min(float(.5*np.mean((rgb[798:845,origin+shift:origin+shift+24].min(2)<=242)[white])+
                    .5*np.mean((rgb[798:845,origin+shift:origin+shift+24].max(2)>=90)[black])) for shift in (-1,0,1)))
            order=np.argsort(scores)
            if scores[order[0]]>.05 or scores[order[1]]-scores[order[0]]<.05:
                raise ValueError('ambiguous ink/outline clock')
            digits.append(int(order[0]));errors.append(scores[order[0]])
        minutes=10*digits[0]+digits[1];seconds=10*digits[2]+digits[3]
        if seconds>=60:raise ValueError('invalid ink clock seconds')
        return (minutes*60+seconds)*1000+(10*digits[4]+digits[5])*10,errors
