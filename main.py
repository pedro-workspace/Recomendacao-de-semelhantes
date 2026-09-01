import argparse
import sys
import numpy as np
from pathlib import Path
from PIL import Image

from recommender import Recommender


def get_query_from_file(path: str) -> Image.Image:
    return Image.open(path)


def get_query_from_webcam() -> Image.Image:
    import cv2

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: não foi possível acessar a webcam.")
        sys.exit(1)

    print("Pressione ESPAÇO para capturar ou ESC para sair.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao ler frame da webcam.")
            break
        cv2.imshow("Webcam - Pressione ESPAÇO para capturar", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == 32:
            cap.release()
            cv2.destroyAllWindows()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame_rgb)

    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)


def display_results(query_image: Image.Image, results: list[dict]):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        import matplotlib.image as mpimg

        n = len(results) + 1
        cols = min(n, 4)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        ax = axes[0, 0]
        ax.imshow(query_image)
        ax.set_title("Consulta", fontsize=14, fontweight="bold", color="blue")
        ax.axis("off")

        for i, r in enumerate(results):
            row = (i + 1) // cols
            col = (i + 1) % cols
            ax = axes[row, col]
            img = mpimg.imread(r["path"])
            ax.imshow(img)
            ax.set_title(
                f"#{i+1} | {r['score']:.3f}\n{r['name']}",
                fontsize=9,
            )
            ax.axis("off")

        for i in range(len(results) + 1, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis("off")

        plt.tight_layout()
        output_path = Path(__file__).parent / "recommendations.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nResultado salvo em: {output_path}")
    except ImportError:
        print("\nmatplotlib não disponível. Exibindo resultados no terminal:\n")
        print(f"Consulta: imagem fornecida")
        print("-" * 50)
        for i, r in enumerate(results):
            print(f"  #{i+1} | Score: {r['score']:.4f} | {r['name']}")
            print(f"         {r['path']}")


def print_results_terminal(results: list[dict]):
    print("\n" + "=" * 60)
    print("  PRODUTOS RECOMENDADOS")
    print("=" * 60)
    for i, r in enumerate(results):
        score_bar = "█" * int(r["score"] * 30)
        print(f"\n  #{i+1}  {r['name']}")
        print(f"      Similaridade: {r['score']:.4f}  {score_bar}")
        print(f"      Caminho: {r['path']}")
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Sistema de Recomendação Visual de Produtos"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Caminho para a imagem de consulta")
    group.add_argument("--webcam", action="store_true", help="Capturar imagem da webcam")
    parser.add_argument("--top", type=int, default=5, help="Número de recomendações (padrão: 5)")
    parser.add_argument("--no-display", action="store_true", help="Não salvar imagem de resultado")

    args = parser.parse_args()

    print("Carregando modelo e índice...")
    rec = Recommender()

    if args.image:
        query = get_query_from_file(args.image)
    else:
        query = get_query_from_webcam()

    print(f"Buscando {args.top} produtos mais similares...")
    results = rec.recommend(query, top_n=args.top)

    print_results_terminal(results)

    if not args.no_display:
        display_results(query, results)


if __name__ == "__main__":
    main()
