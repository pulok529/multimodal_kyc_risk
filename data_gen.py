import os
import numpy as np
from PIL import Image, ImageDraw
import pandas as pd

def generate(out_dir="data", n=4000, seed=3):
    rng=np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    img_dir=os.path.join(out_dir,"images")
    os.makedirs(img_dir, exist_ok=True)

    rows=[]
    for i in range(n):
        # tabular
        age = rng.integers(18, 70)
        doc_quality = rng.beta(3,4)  # 0..1
        mismatch_score = rng.beta(2,6)
        # image: simple stripes + noise as proxy for scan quality
        img = Image.new("L",(64,64), color=255)
        d=ImageDraw.Draw(img)
        # add pattern related to mismatch
        stripes = int(2 + mismatch_score*10)
        for s in range(stripes):
            x = int((s+1)*64/(stripes+1))
            d.line([(x,0),(x,63)], fill=0, width=1)
        # add blur/noise related to quality
        arr=np.array(img).astype(np.float32)
        noise = rng.normal(0, (1-doc_quality)*30, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")

        fn=f"img_{i:05d}.png"
        img.save(os.path.join(img_dir, fn))

        # label: risky if low quality AND high mismatch AND older age a bit
        logit = 2.0*(0.5-doc_quality) + 2.5*(mismatch_score-0.3) + 0.01*(age-40)
        p = 1/(1+np.exp(-logit))
        y = rng.binomial(1, p.clip(0.02,0.95))
        rows.append({"image": fn, "age": age, "doc_quality": doc_quality, "mismatch_score": mismatch_score, "risk": y})

    df=pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir,"labels.csv"), index=False)
    print("Generated", len(df), "samples")

if __name__=="__main__":
    generate()
