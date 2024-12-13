import torch
import matplotlib.pyplot as plt
import argparse
from model import MLPGenerator


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dim", type=int, default=100, help="Dimension of the noise vector"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=512, help="Dimension of the hidden layers"
    )
    parser.add_argument(
        "--output_dim",
        type=int,
        default=28 * 28 * 1,
        help="Dimension of the output image",
    )
    parser.add_argument(
        "--num_samples", type=int, default=4, help="Number of samples to generate"
    )
    parser.add_argument(
        "--num_labels", type=int, default=10, help="Number of unique labels"
    )
    parser.add_argument(
        "--generator_path",
        type=str,
        default="generator.pth",
        help="Path to the generator model",
    )
    return parser


def plot_samples(samples, labels, num_samples):
    samples = samples.view(-1, 28, 28).cpu().numpy()
    fig, axs = plt.subplots(1, num_samples, figsize=(10, 2.5))
    for i in range(num_samples):
        axs[i].imshow(samples[i], cmap="gray")
        axs[i].set_title(f"Label: {labels[i]}")
        axs[i].axis("off")
    plt.show()


if __name__ == "__main__":
    args = get_parser().parse_args()
    input_dim, hidden_dim, output_dim = args.input_dim, args.hidden_dim, args.output_dim
    num_samples, num_labels = args.num_samples, args.num_labels
    generator_path = args.generator_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = MLPGenerator(input_dim, hidden_dim, output_dim).to(device)
    generator.load_state_dict(torch.load(generator_path))
    generator.eval()

    with torch.no_grad():
        while True:
            noise = torch.randn(num_samples, input_dim).to(device)
            labels = torch.randint(0, num_labels, (num_samples,)).to(device)
            generated_samples = generator(noise, labels)

            plot_samples(generated_samples, labels.cpu().numpy(), num_samples)

            cont = input("Generate more samples? (y/n): ").lower()
            if cont != "y":
                break
