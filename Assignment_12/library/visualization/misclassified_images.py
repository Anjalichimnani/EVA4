from imports.imports_eva import *

class misclassfied_images:
    def test_misclassified(model, device, test_loader, nimage = 25):
        model.eval()
        images = []
        preds = []
        actual = []
        #test_loss = 0
        #correct = 0
        count = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                #test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True).view_as(target)  # get the index of the max log-probability
                #correct += pred.eq(target.view_as(pred)).sum().item()
                for a,b,c in zip(data, target, pred):
                    if b!=c:
                        a = a.cpu().numpy()
                        b = b.cpu().numpy()
                        c = c.cpu().numpy()
                        a = (a*0.5)+0.5
                        images.append(a)
                        preds.append(c)
                        actual.append(b)
                        count += 1
                    if count == nimage:
                        return images, actual, preds

    def plot_images(images,actual,preds,nimage=25):
        fig = plt.figure(figsize=(15,18))
        for i in range(nimage):
            ax = fig.add_subplot(5,5,i+1)
            ax.imshow(np.rollaxis(images[i],0,3).squeeze(),cmap='gray')
    
            ax.set_title("Actual: " + str(actual[i]) + " predicted:  " + str(preds[i]))
        plt.show()