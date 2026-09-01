# Visual Product Recommendation System

A content-based image retrieval system that finds visually similar products using deep learning embeddings. Given a query image (file or webcam capture), it returns the most similar products from a pre-indexed catalog based on shape, color, texture, and other visual features.

## How It Works

1. **Feature Extraction** — A ResNet50 CNN (pretrained on ImageNet) processes each image and produces a 2048-dimensional feature vector (embedding).
2. **Indexing** — All catalog images in `products/` are embedded and stored in `index.pkl` for fast lookup.
3. **Similarity Search** — The query image is embedded and compared against the index using cosine similarity. The top-N most similar images are returned.

## Project Structure

```
recommendation_system/
├── products/               # Catalog images (JPG)
├── feature_extractor.py    # ResNet50 feature extraction
├── indexer.py              # Builds the image index (index.pkl)
├── recommender.py          # Similarity search engine
├── main.py                 # CLI entry point (file or webcam)
├── index.pkl               # Precomputed embeddings (generated)
├── recommendations.png     # Visual output (generated)
└── requirements.txt        # Python dependencies
```

## Requirements

- Python 3.10+
- PyTorch + TorchVision
- OpenCV (for webcam capture)
- Pillow
- NumPy
- Matplotlib (for visual output)

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd recommendation_system

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Build the Index

Run once, and again whenever you add/remove images from `products/`:

```bash
python indexer.py
```

This processes all images in `products/` and saves embeddings to `index.pkl`.

### 2. Get Recommendations

**From an image file:**

```bash
python main.py --image path/to/product.jpg --top 10
```

**From webcam:**

```bash
python main.py --webcam --top 5
```

Press **SPACE** to capture a frame, or **ESC** to cancel.

### 3. CLI Options

| Flag | Description |
|------|-------------|
| `--image <path>` | Query with an image file |
| `--webcam` | Query with webcam capture |
| `--top <N>` | Number of recommendations (default: 5) |
| `--no-display` | Skip generating the visual output PNG |

> `--image` and `--webcam` are mutually exclusive.

## Output

- **Terminal** — Ranked list with similarity scores and bar visualization.
- **`recommendations.png`** — Grid image showing the query alongside recommended products.

## Architecture

```
Query Image
    │
    ▼
┌──────────────────┐
│   ResNet50       │  Pretrained on ImageNet, final FC layer removed
│   (2048-dim      │  Output: L2-normalized feature vector
│    embedding)    │
└──────────────────┘
    │
    ▼
┌──────────────────┐
│  Cosine Similarity│  Dot product against all indexed embeddings
│  (matrix mult)    │  (vectors are already L2-normalized)
└──────────────────┘
    │
    ▼
  Top-N Results (ranked by score)
```

## Tech Stack

| Tool | Purpose |
|------|---------|
| [PyTorch](https://pytorch.org/) | Deep learning framework for model inference |
| [TorchVision](https://pytorch.org/vision/) | Pretrained ResNet50 and image transforms |
| [OpenCV](https://opencv.org/) | Webcam capture and image I/O |
| [Pillow](https://python-pillow.org/) | Image loading and preprocessing |
| [NumPy](https://numpy.org/) | Vector operations and similarity computation |
| [Matplotlib](https://matplotlib.org/) | Visual output grid generation |

## Adding New Products

Simply drop `.jpg` images into the `products/` folder and re-run the indexer:

```bash
python indexer.py
```

## License

This project is for educational purposes as part of a DIO challenge.
