def train(num_epochs,latent_size, batch_size, alpha, train_loader, netD, netE, netG,device, optimizerD, optimizerG):
    D_loss = []
    EG_loss = []
    Anomaly_Score = []

    for epoch in range (num_epochs):

        
        loss_D, Loss_R = 0 , 0
        
        netD.train()
        netE.train()
        netG.train()

        for i, data in enumerate(train_loader) :

            images, labels = data
            images = images.to(device)

            z = torch.rand(batch_size,latent_size) - 1
            z = z.to(device)

            Gz = netG(z)
            E_x = netE(images)


            #let's recover the las layer and the output of the discrimintor
            D_E , D_E_F_L = netD(E_x,images)
            D_G , D_G_F_L = netD(z,Gz)
            
            #reconstruction loss 
            loss_R = torch.mean(images - Gz)
            loss_D = criterion(torch.ones_like(D_G_F_L), D_G_F_L)
            
            D_loss.append(loss_D)
            #Score 
            
            A_x = alpha*loss_R + (1-alpha)*loss_D
            
            print(str(epoch) + '  ' + str(loss_R)+'   '+ str(loss_D)+ '   ' +str(A_x))
            optimizerD.zero_grad()
            loss_D.backward(retain_graph=True)
            optimizerD.step()

            optimizerG.zero_grad()
            loss_R.backward()
            optimizerG.step()
        Anomaly_Score.append(A_x)

    return Anomaly_Score