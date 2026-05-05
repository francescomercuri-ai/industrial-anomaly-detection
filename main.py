import os
import numpy as np
from src.config import TRAIN_OK_DIR, TEST_DEFECT_DIR
from src.model import PatchCoreExtractor
from src.evaluate import calculate_image_level_auroc

def main():
    # Inizializza il modello (metti 'cuda' se sei su GPU)
    detector = PatchCoreExtractor(device='cpu')
    
    # Carico di immagini sane per costruzione Memory Bank
    ok_files = [os.path.join(TRAIN_OK_DIR, f) for f in os.listdir(TRAIN_OK_DIR) if f.endswith(('.jpeg', '.jpg', '.png'))][:50]
    detector.fit(ok_files)
    # Creazione test set (30 immagini OK, 30 immagini Difettose)
    test_ok = [os.path.join(TRAIN_OK_DIR, f) for f in os.listdir(TRAIN_OK_DIR) if f.endswith(('.jpeg', '.jpg', '.png'))][50:80]
    test_defects = [os.path.join(TEST_DEFECT_DIR, f) for f in os.listdir(TEST_DEFECT_DIR) if f.endswith(('.jpeg', '.jpg', '.png'))][:30]
    test_images = test_ok + test_defects
    y_true = [0]*len(test_ok) + [1]*len(test_defects)
    y_scores = []
    
    print("Valutazione immagini di test in corso...")
    for img_path in test_images:
        anomaly_map = detector.predict(img_path)
        # IMAGE-LEVEL SCORE
        image_score = np.max(anomaly_map)
        y_scores.append(image_score)
        
    # Calcolo metriche finali
    auroc = calculate_image_level_auroc(y_true, y_scores)
    print("\n" + "="*40)
    print("RISULTATI VALUTAZIONE:")
    print(f"Image-Level AUROC: {auroc:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()