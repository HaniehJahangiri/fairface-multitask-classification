model, loss_fn, optimizer, schedular = get_model()

def race_Accuracy(outputs,targets):
    _,preds = torch.max(outputs,dim=1)
    correct = (preds==targets).float().sum()
    return correct.item()

def gender_Accuracy(outputs,targets):
    preds = (outputs>.5).float()
    preds = preds.squeeze()
    correct = (preds==targets).float().sum()
    return correct.item()

def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss_epoch = 0.0
    raceAccuracy = 0.0
    genderAccuracy = 0.0
    age_mae = 0.0
    for image, age, gender, race in dataloader:
        optimizer.zero_grad()
        image, age, gender, race = image.to(device), age.to(device), gender.to(device), race.to(device)
        pred_age, pred_gender, pred_race = model(image)
        total_loss_batch, age_loss, gender_loss, race_loss = criterion(pred_age, pred_gender, pred_race, age, gender, race)
        total_loss_batch.backward()
        optimizer.step()
        total_loss_epoch += total_loss_batch.item()
        genderAccuracy += gender_Accuracy(pred_gender,gender)
        raceAccuracy += race_Accuracy(pred_race,race)
        age_mae += torch.abs(age - pred_age.squeeze()).float().sum().item()
    return total_loss_epoch/len(dataloader), age_mae/(32*len(dataloader)), raceAccuracy/(32*len(dataloader)), genderAccuracy/(32*len(dataloader))

def validation(model, dataloader, criterion):
    model.eval()
    total_loss_epoch =0.0
    raceAccuracy = 0.0
    genderAccuracy = 0.0
    age_mae = 0.0
    with torch.no_grad():
        for image, age, gender, race in dataloader:
            image, age, gender, race = image.to(device), age.to(device), gender.to(device), race.to(device)
            pred_age, pred_gender, pred_race = model(image)
            total_loss_batch, age_loss, gender_loss, race_loss = criterion(pred_age, pred_gender, pred_race, age, gender, race)
            total_loss_epoch += total_loss_batch.item()
            genderAccuracy += gender_Accuracy(pred_gender,gender)
            raceAccuracy += race_Accuracy(pred_race,race)
            age_mae += torch.abs(age - pred_age.squeeze()).float().sum().item()
    return total_loss_epoch/len(dataloader), age_mae/(32*len(dataloader)), raceAccuracy/(32*len(dataloader)), genderAccuracy/(32*len(dataloader))