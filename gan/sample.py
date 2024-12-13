import torch
import matplotlib.pyplot as plt
import argparse
from model import MLPGenerator


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--output_dim", type=int, default=28 * 28 * 1)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--generator_path", type=str, default="generator.pth")
    return parser


def plot_samples(samples, num_samples):
    samples = samples.view(-1, 28, 28).cpu().numpy()
    fig, axs = plt.subplots(1, num_samples, figsize=(10, 2.5))
    for i in range(num_samples):
        axs[i].imshow(samples[i], cmap="gray")
        axs[i].axis("off")
    plt.show()


if __name__ == "__main__":
    args = get_parser().parse_args()
    input_dim, hidden_dim, output_dim = args.input_dim, args.hidden_dim, args.output_dim
    num_samples = args.num_samples
    generator_path = args.generator_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = MLPGenerator(input_dim, hidden_dim, output_dim).to(device)
    generator.load_state_dict(torch.load(generator_path))
    generator.eval()
    with torch.no_grad():
        while True:
            noise = torch.randn(num_samples, input_dim).to(device)
            generated_samples = generator(noise)
            plot_samples(generated_samples, num_samples)
            cont = input("Generate more samples? (y/n): ").lower()
            if cont != "y":
                break
