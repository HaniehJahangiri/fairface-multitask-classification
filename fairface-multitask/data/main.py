import torch
from models.model import get_model
from training.train import train_epoch, validate_epoch
from utils.helpers import plot_training_curves, print_epoch_results

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get model
    model, criterion, optimizer, scheduler = get_model(device)
    
    #Dataset
    class AgeGenderRace_Dataset(Dataset):
        def __init__(self,df):
            self.df = df
            self.transform = transforms.Compose([transforms.ToPILImage(),
                                              transforms.Resize((224,224)),
                                              transforms.ToTensor(),
                                            transforms.Normalize(mean= [0.485, 0.456, 0.406],
                                                                std= [0.229, 0.224, 0.225])])
            self.race_mapping = {'Latino_Hispanic': 0,
                            'East Asian': 1,
                            'Indian': 2,
                            'Middle Eastern': 3,
                            'Black': 4,
                            'Southeast Asian': 5,
                            'White': 6}
        def __len__(self):
            return len(self.df)
        def __getitem__(self, idx):
            f = self.df.iloc[idx].squeeze()
            file = f.file
            age = float(f.age/80)
            age = torch.tensor(age,dtype=torch.float32)
            gender = float(f.gender == 'Female')
            gender = torch.tensor(gender,dtype=torch.float32)
            race = self.race_mapping[f.race]
            race = torch.tensor(race,dtype=torch.long)
            image = cv2.imread(file)
            if image is None:
                image = np.zeros((224,224,3),dtype=np.unint8)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.transform(image)
            return image,age,gender,race
    train_dataset = AgeGenderRace_Dataset(train_df)
    test_dataset = AgeGenderRace_Dataset(test_df)

    #DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size= 32, shuffle= True, drop_last= True,num_workers=0,pin_memory= False,persistent_workers=False)
    test_dataloader = DataLoader(test_dataset, batch_size= 32, shuffle= False, num_workers=0,pin_memory= False,persistent_workers=False)

    n_epochs = 10
    train_losses , valid_losses = [] , []
    train_race_accuracies, train_gender_accuracies, train_age_maes = [] , [] ,[]
    valid_race_accuracies, valid_gender_accuracies, valid_age_maes = [] , [] ,[]

    for epoch in range(n_epochs):
        train_loss, train_age_mae, train_race_accuracy, train_gender_accuracy = train(model, train_dataloader, optimizer, loss_fn)
        train_losses.append(train_loss)
        train_race_accuracies.append(train_race_accuracy)
        train_age_maes.append(train_age_mae)
        train_gender_accuracies.append(train_gender_accuracy)
    
        test_loss, test_age_mae, test_race_accuracy, test_gender_accuracy = validation(model, test_dataloader, loss_fn)
        valid_losses.append(test_loss)
        valid_age_maes.append(test_age_mae)
        valid_gender_accuracies.append(test_gender_accuracy)
        valid_race_accuracies.append(test_race_accuracy)
    
    print(f'Epoch:{epoch+1}/{n_epochs}, Train Loss:{train_loss:.4f}, Validation Loss:{test_loss:.4f},\n\t',
          f'Train gender Accuracy: {train_gender_accuracy:.4f}, Validation gender Accuracy: {test_gender_accuracy:.4f},\n\t',
          f'Train Race Accuracy:{train_race_accuracy:.4f}, Validation Race Accuracy:{test_race_accuracy:.4f},\n\t',
          f'Train Age MAE:{train_age_mae:.4f}, Validation Age Mae:{test_age_mae:.4f},\n\t')
    #Plot results
    plt.figure(figsize=(16,8))

    plt.subplot(2,2,1)
    plt.plot(train_losses,label='Train Loss')
    plt.plot(valid_losses,label='Test Loss')
    plt.title('Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.tight_layout(pad= 3)
    plt.legend()

    #Plotting Gender Accuracies
    plt.subplot(2,2,2)
    plt.plot(train_gender_accuracies,label='Train Gender Accuracy')
    plt.plot(valid_gender_accuracies,label='Test Gender Accuracy')
    plt.title('Gender Accuracies')
    plt.xlabel('Epochs')
    plt.ylabel('Gender Accuracy')
    plt.tight_layout(pad= 3)
    plt.legend()

    #Plotting Race Accuracies
    plt.subplot(2,2,3)
    plt.plot(train_race_accuracies,label='Train Race Accuracy')
    plt.plot(valid_race_accuracies,label='Test Race Accuracy')
    plt.title('Race Accuracies')
    plt.xlabel('Epochs')
    plt.ylabel('Race Accuracy')
    plt.tight_layout(pad= 3)
    plt.legend()

    #Plotting Age Mean Absolute Error
    plt.subplot(2,2,4)
    plt.plot(train_age_maes,label='Train Age MAE')
    plt.plot(valid_age_maes,label='Test Age MAE')
    plt.title('Age MAEs')
    plt.xlabel('Epochs')
    plt.ylabel('Age MAE')
    plt.tight_layout(pad= 3)
    plt.legend()

    plt.show()

if __name__ == "__main__":
    main()