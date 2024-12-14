from model import MLPDiscriminator, MLPGenerator
import torch
import torch.optim as optim
import argparse
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dim", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--train_gen_each", type=int, default=2)
    parser.add_argument("--preload", action="store_true", default=False)
    return parser


dims = {
    "MNIST": 28 * 28 * 1,
    "CIFAR10": 32 * 32 * 3,
}

if __name__ == "__main__":
    # Setup
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Arguments
    args = get_parser().parse_args()
    input_dim, hidden_dim = args.input_dim, args.hidden_dim
    num_epochs, batch_size, lr, train_gen_each = (
        args.num_epochs,
        args.batch_size,
        args.lr,
        args.train_gen_each,
    )

    output_dim = dims["MNIST"]

    # Models and optimizers
    generator = MLPGenerator(input_dim, hidden_dim, output_dim).to(device)
    discriminator = MLPDiscriminator(output_dim, hidden_dim, 1).to(device)
    generator_optimizer = optim.Adam(generator.parameters(), lr=lr)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

    if args.preload:
        generator.load_state_dict(torch.load("generator.pth"))
        discriminator.load_state_dict(torch.load("discriminator.pth"))

    # Data (MNIST)
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Lambda(lambda x: x.view(-1).to(device)),
        ]
    )

    mnist_train = datasets.MNIST(
        root="data", train=True, transform=transform, download=True
    )
    train_dataloader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    mnist_test = datasets.MNIST(
        root="data", train=False, transform=transform, download=True
    )
    test_dataloader = DataLoader(mnist_test, batch_size=batch_size * 8, shuffle=True)

    step = 0

    # DISCRIMINATOR LOSS
    def loss_real_disc(disc_outs):
        return torch.log(disc_outs + 1e-8).mean()  # maximize log(D(x))

    def loss_fake_disc(disc_outs):
        return torch.log(1 - disc_outs + 1e-8).mean()  # maximize log(1 - D(G(z)))

    # GENERATOR LOSS
    def loss_gen(disc_outs):
        return torch.log(disc_outs + 1e-8).mean()  # maximize log(D(G(z)))

    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()

        for real_data, _ in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
            noise = torch.randn(real_data.size(0), input_dim).to(device)
            if step % (train_gen_each + 1) != 0:
                # Train discriminator
                discriminator_optimizer.zero_grad()
                # real data -> probab = 1
                loss1 = loss_real_disc(discriminator(real_data))
                # fake data -> probab = 0
                loss2 = loss_fake_disc(discriminator(generator(noise).detach()))
                loss = loss1 + loss2
                loss = -loss
                loss.backward()
                discriminator_optimizer.step()
            else:
                # Train generator
                generator_optimizer.zero_grad()
                # try to make discriminator think probab = 1 (we are generating real data)
                loss = loss_gen(discriminator(generator(noise)))
                loss = -loss
                loss.backward()
                generator_optimizer.step()
            step += 1

        # test
        generator.eval()
        discriminator.eval()
        with torch.no_grad():
            discriminator_loss_sum = 0
            generator_loss_sum = 0
            for real_data, _ in test_dataloader:
                noise = torch.randn(real_data.size(0), input_dim).to(device)
                discriminator_loss_sum += loss_real_disc(
                    discriminator(real_data)
                ) + loss_fake_disc(discriminator(generator(noise).detach()))

                generator_loss_sum += loss_gen(discriminator(generator(noise)))

            generator_loss_sum /= len(test_dataloader)
            discriminator_loss_sum /= len(test_dataloader)

        print(
            f"Epoch {epoch}, Discriminator loss: {discriminator_loss_sum}, Generator loss: {generator_loss_sum}"
        )
        torch.save(generator.state_dict(), "generator.pth")
        torch.save(discriminator.state_dict(), "discriminator.pth")
