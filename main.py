import os
import numpy as np
from src.config import TRAIN_OK_DIR, TEST_DEFECT_DIR
from src.model import PatchCoreExtractor
from src.evaluate import calculate_image_level_auroc

def main():
    # Inizializza il modello (metti 'cuda' se sei su GPU)
    detector = PatchCoreExtractor(device='cpu')
    
    # 1. Carica le immagini OK per addestrare la Memory Bank
    # Per non esplodere la RAM, ne prendiamo solo 50 per fare un test
    ok_files = [os.path.join(TRAIN_OK_DIR, f) for f in os.listdir(TRAIN_OK_DIR) if f.endswith(('.jpeg', '.jpg', '.png'))][:50]
    
    detector.fit(ok_files)
    
    # 2. Creiamo un Test Set (30 immagini OK, 30 immagini Difettose)
    # Assicurati di prendere immagini OK DIVERSE da quelle usate nel fit!
    test_ok = [os.path.join(TRAIN_OK_DIR, f) for f in os.listdir(TRAIN_OK_DIR) if f.endswith(('.jpeg', '.jpg', '.png'))][50:80]
    test_defects = [os.path.join(TEST_DEFECT_DIR, f) for f in os.listdir(TEST_DEFECT_DIR) if f.endswith(('.jpeg', '.jpg', '.png'))][:30]
    
    test_images = test_ok + test_defects
    # Etichette: 0 per OK, 1 per Difettoso
    y_true = [0]*len(test_ok) + [1]*len(test_defects)
    
    y_scores = []
    
    print("Valutazione immagini di test in corso...")
    for img_path in test_images:
        anomaly_map = detector.predict(img_path)
        
        # IMAGE-LEVEL SCORE: Prendo il valore massimo della mappa di anomalia
        # Se c'è anche solo un graffio, il valore massimo sarà altissimo.
        image_score = np.max(anomaly_map)
        y_scores.append(image_score)
        
    # 3. Calcolo metriche finali
    auroc = calculate_image_level_auroc(y_true, y_scores)
    
    print("\n" + "="*40)
    print("RISULTATI VALUTAZIONE:")
    print(f"Image-Level AUROC: {auroc:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()