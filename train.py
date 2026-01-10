import os, argparse
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class KYCDataset(Dataset):
    def __init__(self, df, root):
        self.df=df.reset_index(drop=True)
        self.root=root
        self.tf = transforms.Compose([
            transforms.ToTensor(),
        ])
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        r=self.df.iloc[idx]
        img = Image.open(os.path.join(self.root,"images", r["image"])).convert("L")
        x_img = self.tf(img)  # (1,64,64)
        x_tab = torch.tensor([r["age"]/70.0, r["doc_quality"], r["mismatch_score"]], dtype=torch.float32)
        y = torch.tensor(r["risk"], dtype=torch.float32)
        return x_img, x_tab, y

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1,8,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self.img_head = nn.Linear(16*16*16, 32)
        self.tab = nn.Sequential(nn.Linear(3,16), nn.ReLU(), nn.Linear(16,16), nn.ReLU())
        self.out = nn.Linear(32+16, 1)

    def forward(self, x_img, x_tab):
        z = self.cnn(x_img)
        z = torch.relu(self.img_head(z))
        t = self.tab(x_tab)
        h = torch.cat([z,t], dim=1)
        return self.out(h).squeeze(1)

def train_epoch(model, loader, opt, device):
    model.train()
    loss_fn=nn.BCEWithLogitsLoss()
    total=0; n=0
    for x_img, x_tab, y in loader:
        x_img=x_img.to(device); x_tab=x_tab.to(device); y=y.to(device)
        opt.zero_grad()
        logits=model(x_img,x_tab)
        loss=loss_fn(logits, y)
        loss.backward()
        opt.step()
        total+=loss.item()*len(y); n+=len(y)
    return total/n

@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    ys=[]; ps=[]
    for x_img, x_tab, y in loader:
        x_img=x_img.to(device); x_tab=x_tab.to(device)
        logits=model(x_img,x_tab)
        p=torch.sigmoid(logits).cpu().numpy()
        ps.extend(p.tolist()); ys.extend(y.numpy().tolist())
    auc=roc_auc_score(ys, ps)
    acc=accuracy_score(ys, (np.array(ps)>=0.5).astype(int))
    return auc, acc

def main():
    if not os.path.exists("data/labels.csv"):
        from data_gen import generate
        generate()
    df=pd.read_csv("data/labels.csv")
    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["risk"])

    device="cuda" if torch.cuda.is_available() else "cpu"
    model=Model().to(device)
    opt=torch.optim.Adam(model.parameters(), lr=1e-3)

    train_loader=DataLoader(KYCDataset(train_df,"data"), batch_size=64, shuffle=True)
    valid_loader=DataLoader(KYCDataset(valid_df,"data"), batch_size=128, shuffle=False)

    for epoch in range(1,6):
        loss=train_epoch(model, train_loader, opt, device)
        auc, acc = eval_model(model, valid_loader, device)
        print(f"epoch {epoch}: loss={loss:.4f}  AUC={auc:.3f}  ACC={acc:.3f}")

    torch.save(model.state_dict(), "model.pt")
    print("Saved model.pt")

if __name__=="__main__":
    main()
