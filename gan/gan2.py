
from gan1 import *

if __name__ == '__main__':
    from numpy import dot
    import joblib
    from sklearn.preprocessing import StandardScaler
    from utils.data_read import credit_card
    data, data_cols, label_cols = credit_card()
    stdscaler = StandardScaler()
    stdscaler.fit(data[data_cols])
    data[data_cols] = stdscaler.transform(data[data_cols])
    joblib.dump(stdscaler, 'data_scale2.jbl')

    train = TensorDataset(torch.tensor(data[data_cols].values, dtype=torch.float64),)
    train_loader = DataLoader(train, batch_size=16, shuffle=True)

    # Set your parameters
    criterion = nn.BCEWithLogitsLoss()
    n_epochs = 10
    z_dim = 10
    hidden_dim = 20
    display_step = 500
    batch_size = 16
    lr = 0.00001
    device = 'cpu'
    PATH = "gen_model2.pt"
    gen = Generator(z_dim,im_dim=30).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
    disc = Discriminator(im_dim=30).to(device)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)
    cur_step = 0
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    test_generator = True  # Whether the generator should be tested
    gen_loss = False
    error = False
    for epoch in range(n_epochs):

        # Dataloader returns the batches
        for real in tqdm(train_loader):
            real = real[0]
            cur_batch_size = len(real)

            # Flatten the batch of real images from the dataset
            real = real.view(cur_batch_size, -1).to(device)

            ### Update discriminator ###
            # Zero out the gradients before backpropagation
            disc_opt.zero_grad()

            # Calculate discriminator loss
            disc_loss = get_disc_loss(gen, disc, criterion, real, cur_batch_size, z_dim, device)

            # Update gradients
            disc_loss.backward(retain_graph=True)

            # Update optimizer
            disc_opt.step()

            # For testing purposes, to keep track of the generator weights
            if test_generator:
                old_generator_weights = gen.gen[0][0].weight.detach().clone()

            ### Update generator ###
            #     Hint: This code will look a lot like the discriminator updates!
            #     These are the steps you will need to complete:
            #       1) Zero out the gradients.
            #       2) Calculate the generator loss, assigning it to gen_loss.
            #       3) Backprop through the generator: update the gradients and optimizer.
            #### START CODE HERE ####
            gen_opt.zero_grad()
            gen_loss = get_gen_loss(gen, disc, criterion, cur_batch_size, z_dim, device)
            gen_loss.backward()
            gen_opt.step()
            #### END CODE HERE ####

            # For testing purposes, to check that your code changes the generator weights
            if test_generator:
                try:
                    assert lr > 0.0000002 or (gen.gen[0][0].weight.grad.abs().max() < 0.0005 and epoch == 0)
                    assert torch.any(gen.gen[0][0].weight.detach().clone() != old_generator_weights)
                except:
                    error = True
                    print("Runtime tests have failed")

            # Keep track of the average discriminator loss
            mean_discriminator_loss += disc_loss.item() / display_step

            # Keep track of the average generator loss
            mean_generator_loss += gen_loss.item() / display_step

            ### Visualization code ###
            if cur_step % display_step == 0 and cur_step > 0:
                print(
                    f"Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
                fake_noise = get_noise(cur_batch_size, z_dim, device=device)
                fake = gen(fake_noise)
                mean_generator_loss = 0
                mean_discriminator_loss = 0
            cur_step += 1
        # Save the model
        torch.save({
            'epoch': n_epochs,
            'model_state_dict': gen.state_dict(),
            'optimizer_state_dict': gen_opt.state_dict(),
            'loss': gen_loss,
        }, PATH)
