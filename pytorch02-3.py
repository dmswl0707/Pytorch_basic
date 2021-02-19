###veridation

correct=0
total=0

with torch.no_grad(): #테스트를 하는데 기울기 계산하지 않음
    for image, label in test_loader:
        x=image.to(device)
        y=label.to(device)

        output=model.forward(x)
        _,output_index=torch.max(output,1)
