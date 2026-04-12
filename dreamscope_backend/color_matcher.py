from colorthief import ColorThief
import os
import pandas as pd
from PIL import Image
import matplotlib
from soupsieve import closest
matplotlib.use('Agg')
import matplotlib.pyplot as plt

image_dir = "dreamscope_backend/data/abstract_art_512"

def build_color_index(n=None):
    images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    if n:
        images = images[:n]
    
    records = []
    for i, img in enumerate(images):
        path = os.path.join(image_dir, img)
        try:
            ct = ColorThief(path)
            r, g, b = ct.get_color(quality=1)
            records.append({"filename": img, "r": r, "g": g, "b": b})
        except Exception as e:
            print(f"skipping {img}: {e}")
        
        if i % 100 == 0:
            print(f"{i}/{len(images)} done")
    
    df = pd.DataFrame(records)
    df.to_csv("dreamscope_backend/data/image_colors.csv", index=False)
    print(f"saved {len(df)} images")


EMOTION_COLORS = {
    "admiration":    (255, 200, 100),
    "amusement":     (255, 220, 50),
    "anger":         (180, 30,  30),
    "annoyance":     (160, 60,  40),
    "approval":      (100, 180, 100),
    "caring":        (220, 130, 180),
    "confusion":     (150, 100, 180),
    "curiosity":     (50,  150, 200),
    "desire":        (200, 50,  100),
    "disappointment":(100, 100, 120),
    "disapproval":   (120, 80,  60),
    "disgust":       (60,  90,  40),
    "embarrassment": (220, 120, 100),
    "excitement":    (255, 140, 0),
    "fear":          (20,  20,  40),
    "gratitude":     (180, 220, 150),
    "grief":         (40,  40,  80),
    "joy":           (255, 220, 50),
    "love":          (220, 80,  120),
    "nervousness":   (180, 160, 80),
    "optimism":      (255, 165, 0),
    "pride":         (100, 50,  150),
    "realization":   (100, 180, 200),
    "relief":        (160, 210, 190),
    "remorse":       (80,  60,  100),
    "sadness":       (70,  100, 160),
    "surprise":      (200, 100, 220),
    "neutral":       (180, 180, 180),
}


def color_distance(rgb1, rgb2):
    return sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)) ** 0.5


def match_images_by_emotion(top_emotion, n=3):
    df_colors = pd.read_csv("dreamscope_backend/data/image_colors.csv")
    target = EMOTION_COLORS.get(top_emotion, (180, 180, 180))
    df_colors["distance"] = df_colors.apply(
        lambda row: color_distance((row["r"], row["g"], row["b"]), target), axis=1
    )
    # take top 20 closest, pick 3 randomly
    closest = df_colors.nsmallest(n, "distance")
    return closest["filename"].tolist()


if __name__ == "__main__":
    # build_color_index(n=1000)  # run once, then comment out

    # test matching
    test_emotions = ["joy", "fear", "sadness", "anger", "surprise"]
    for emotion in test_emotions:
        images = match_images_by_emotion(emotion, n=3)
        print(f"\n{emotion}:")
        for img in images:
            print(f"  {img}")
    def show_matches(emotions_to_test):
        fig, axes = plt.subplots(len(emotions_to_test), 3, figsize=(12, 4 * len(emotions_to_test)))
        for row, emotion in enumerate(emotions_to_test):
            images = match_images_by_emotion(emotion, n=3)
            for col, img_name in enumerate(images):
                img_path = os.path.join(image_dir, img_name)
                img = Image.open(img_path)
                axes[row, col].imshow(img)
                axes[row, col].set_title(f"{emotion}" if col == 0 else "")
                axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig("dreamscope_backend/data/color_match_test.png")
    print("saved to color_match_test.png")

    show_matches(["joy", "fear", "sadness", "anger", "surprise"])
    
