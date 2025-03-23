import torchvision.transforms as tt

def transform():
    train_tfms = tt.Compose([
                         tt.Grayscale(num_output_channels=3), # Pictures black and white
                         tt.Resize([128, 128]),
                         tt.ToTensor(),                      # Cast to tensor
                         ])                      
    return train_tfms
