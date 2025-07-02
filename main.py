import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# --- モデル定義 ---
class Encoder(nn.Module):
    def __init__(self, latent_dim=3, num_classes=10):
        super().__init__()
        self.label_embedding = nn.Linear(num_classes, 16)
        self.fc_hidden = nn.Linear(28 * 28 + 16, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, image, label):
        flattened_image = image.view(image.size(0), -1)
        label_one_hot = F.one_hot(label, num_classes=10).float()
        label_embedding = F.relu(self.label_embedding(label_one_hot))
        concatenated_input = torch.cat([flattened_image, label_embedding], dim=1)
        hidden = F.relu(self.fc_hidden(concatenated_input))
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=3, num_classes=10):
        super().__init__()
        self.label_embedding = nn.Linear(num_classes, 16)
        self.fc_hidden = nn.Linear(latent_dim + 16, 128)
        self.fc_out = nn.Linear(128, 28 * 28)

    def forward(self, latent_vector, label):
        label_one_hot = F.one_hot(label, num_classes=10).float()
        label_embedding = F.relu(self.label_embedding(label_one_hot))
        concatenated = torch.cat([latent_vector, label_embedding], dim=1)
        hidden = F.relu(self.fc_hidden(concatenated))
        output = torch.sigmoid(self.fc_out(hidden))
        return output.view(-1, 1, 28, 28)

class CVAE(nn.Module):
    def __init__(self, latent_dim=3, num_classes=10):
        super().__init__()
        self.encoder = Encoder(latent_dim, num_classes)
        self.decoder = Decoder(latent_dim, num_classes)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y):
        mu, logvar = self.encoder(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z, y), mu, logvar

# --- Streamlit アプリ ---
st.title("🧠 CVAE 画像生成アプリ")
st.markdown("数字と潜在変数を指定して手書き数字画像を生成します。")

# モデルロード
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 3
model = CVAE(latent_dim=latent_dim).to(device)
model.load_state_dict(torch.load("cvae.pth", map_location=device))
model.eval()

# ユーザー入力
digit = st.selectbox("生成する数字 (0〜9)", list(range(10)))
z_inputs = []
for i in range(latent_dim):
    z_inputs.append(st.slider(f"潜在変数 z[{i}]", -3.0, 3.0, 0.0, step=0.1))

# 生成ボタン
if st.button("生成する"):
    z = torch.tensor([z_inputs], dtype=torch.float32, device=device)
    label = torch.tensor([digit], dtype=torch.long, device=device)

    with torch.no_grad():
        output = model.decoder(z, label)
    
    img = output.squeeze().cpu().numpy()
    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray")
    ax.axis("off")
    st.pyplot(fig)
