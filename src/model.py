import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors
import numpy as np
from PIL import Image

class PatchCoreExtractor:
    def __init__(self, device='cpu'): # Su PC usiamo CPU di default
        self.device = torch.device(device)
        # Carica ResNet18
        weights = models.ResNet18_Weights.IMAGENETIK_V1
        self.model = models.resnet18(weights=weights).to(self.device)
        self.model.eval()
        
        self.features = None
        self.model.layer2.register_forward_hook(self._hook_fn)
        
        self.knn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
        self.memory_bank = None
        
        self.preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _hook_fn(self, module, input, output):
        self.features = output.detach()

    def get_features(self, image_path):
        img = Image.open(image_path).convert('RGB')
        tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            self.model(tensor)
        
        feats = self.features.squeeze() # [128, 32, 32]
        # Flatten: [128, 1024] -> trasposto: [1024, 128]
        return feats.view(feats.size(0), -1).t().cpu().numpy()

    def fit(self, image_paths):
        print(f"Costruzione Memory Bank con {len(image_paths)} immagini...")
        bank = []
        for path in image_paths:
            bank.append(self.get_features(path))
        
        self.memory_bank = np.vstack(bank)
        self.knn.fit(self.memory_bank)
        print(f"Memory Bank pronta. Dimensioni: {self.memory_bank.shape}")

    def predict(self, image_path):
        """Ritorna la mappa delle distanze (anomalie)"""
        test_feats = self.get_features(image_path)
        distances, _ = self.knn.kneighbors(test_feats)
        return distances.reshape(32, 32)