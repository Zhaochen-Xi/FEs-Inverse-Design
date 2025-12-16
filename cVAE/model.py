import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import List
from torch.autograd import Variable


class Encoder(nn.Module):
    def __init__(self, in_dims, h_dims: List):
        super(Encoder, self).__init__()
        encoder_list = []
        for h_dim in h_dims:
            encoder_list.append(
                nn.Sequential(
                    nn.Conv2d(in_dims, out_channels=h_dim, 
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU(inplace=True)
                )
            )
            in_dims = h_dim
        self.encoder = nn.Sequential(*encoder_list)

    def forward(self, x):
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)
        return result

class Latent(nn.Module):
    def __init__(self, in_dims, latent_dims):
        super(Latent, self).__init__()
        self.fc_mu = nn.Linear(in_dims, latent_dims)
        self.fc_var = nn.Linear(in_dims, latent_dims)

    def forward(self, x):
        return [self.fc_mu(x), self.fc_var(x)]

class Decoder(nn.Module):
    def __init__(self, latent_dims, h_dims: List):
        super(Decoder, self).__init__()
        self.h_dims = h_dims
        self.decoder_input = nn.Linear(latent_dims, h_dims[0] * 4)

        decoder_list = []
        for i in range(len(h_dims) - 1):
            decoder_list.append(
                nn.Sequential(
                    nn.ConvTranspose2d(h_dims[i], h_dims[i+1],
                                       kernel_size=3, stride=2,
                                       padding=1, output_padding=1),
                    nn.BatchNorm2d(h_dims[i+1]),
                    nn.ReLU(inplace=True)
                )
            )
        self.decoder = nn.Sequential(*decoder_list)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(h_dims[-1], h_dims[-1],
                                                  kernel_size=3, stride=2,
                                                  padding=1, output_padding=1),
                            nn.BatchNorm2d(h_dims[-1]),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(h_dims[-1], out_channels=3,
                                      kernel_size=3, padding=1),
                            nn.Tanh())

    def forward(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, self.h_dims[0], 2, 2) 
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

class CVAE(nn.Module):
    def __init__(self, in_channels: int, latent_dims: int, num_classes: int, img_size: int = 128, hidden_dims: List = None, **kwargs) -> None:
        super(CVAE, self).__init__()
        
        self.latent_dims = latent_dims
        self.img_size = img_size
        
        if hidden_dims is None:
            hidden_dims = [64, 64, 256, 256, 512, 512]
            
        # Parameterized embedding size based on image size
        self.embed_classes = nn.Linear(in_features=num_classes, out_features=img_size * img_size)
        self.embed_data = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        
        # Encoder input channels + 1 for the embedded class channel
        self.encoder = Encoder(in_channels + 1, hidden_dims)
        
        # Calculate bottleneck size dynamically or assume standard reduction
        # (Assuming hidden_dims[-1] * 4 from original code logic)
        self.latent = Latent(hidden_dims[-1] * 4, latent_dims)
        
        hidden_dims.reverse()
        self.decoder = Decoder(latent_dims=latent_dims + num_classes, h_dims=hidden_dims)
        self.loss = torch.nn.MSELoss()

    def forward(self, input: Tensor, classes: Tensor):
        classes = classes.float()
        embedded_class = self.embed_classes(classes)
        # Reshape based on dynamic img_size
        embedded_class = embedded_class.view(-1, self.img_size, self.img_size).unsqueeze(1)
        
        embedded_data = self.embed_data(input)
        embedded_input = torch.cat([embedded_data, embedded_class], dim=1)
        
        z = self.encoder(embedded_input)
        mu, log_var = self.latent(z)
        z = self.reparameterize(mu, log_var)
        z = torch.cat([z, classes], dim=1)
        
        return [self.decoder(z), input, mu, log_var]

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def loss_function(self, *output, **kwargs):
        recons = output[0]
        orig = output[1]
        mu = output[2]
        log_var = output[3]
        kld_weight = kwargs['weight']

        recons_loss = self.loss(recons, orig)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {'Loss': loss, 'Recon_Loss': recons_loss, 'Kld_Loss': kld_loss}